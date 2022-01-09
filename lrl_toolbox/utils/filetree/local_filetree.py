import datetime
import functools
import json
import math
import os
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

import fasteners  # Cross platform locks
import pandas as pd
import tqdm
import yaml  # type: ignore
from pydantic import BaseModel, Field, validator
from pydantic.typing import Literal

from lrl_toolbox import logger
from lrl_toolbox.utils.filetree.exceptions import InvalidRootDirException, InvalidTreeOperationException


###
# Auxiliary classes
#  Config class
#  Data Entry
###
class LocalFileTreeConfig(BaseModel):

    file_count: int = Field(0, description="")
    tree_depth: int = Field(2, description="")

    leaf_depth: int = Field(7, description="")

    file_format: Literal["feather", "csv", "json", "yaml", "pickle"] = Field(True, description="")

    has_metadata: bool = Field(True, description="")

    # last_update: datetime.datetime = Field(default_factory= lambda: datetime.datetime.now())

    class Config:
        underscore_attrs_are_private = True
        validate_assignment = True
        allow_mutation = False


class TreeDataEntry(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    BINARY_FORMATS: ClassVar[Set[str]] = {"feather", "pickle"}

    DATA_FILE_PARSERS: ClassVar[Dict[str, Any]] = {
        "csv": pd.read_csv,
        "feather": pd.read_feather,
        "json": json.load,
        "yaml": yaml.load,
        "pickle": pickle.load,
    }

    DATA_FILE_WRITERS: ClassVar[Dict[str, Any]] = {
        "csv": functools.partial(pd.DataFrame.to_csv, index=False),
        "feather": pd.DataFrame.to_feather,
        "json": json.dump,
        "yaml": yaml.dump,
        "pickle": functools.partial(pickle.dump, protocol=pickle.HIGHEST_PROTOCOL),
    }

    data: Union[pd.DataFrame, Dict[Any, Any], Any]
    metadata: Optional[Dict[str, Any]]

    @validator("data", pre=True)
    def load_data(cls, value: Any):
        if isinstance(value, str) or isinstance(value, Path):
            path = str(value)
            # Get file extension
            filename, file_extension = os.path.splitext(path)
            # Remove the starting dot from the file extension
            file_extension = file_extension.replace(".", "")
            parser = cls.DATA_FILE_PARSERS[file_extension]
            is_binary = file_extension in cls.BINARY_FORMATS
            mode = "rb" if is_binary else "r"
            encoding = None if is_binary else "utf-8"
            with open(value, mode, encoding=encoding) as f:
                return parser(f)
        else:
            return value

    @validator("metadata", pre=True)
    def load_metadata(cls, value: Any):
        if value is None:
            return value
        elif isinstance(value, dict):
            return value
        else:
            with open(value, "r", encoding="utf-8") as f:
                value = json.load(f)
            return value

    def write_data(self, path: Union[str, Path], format: str):
        writer = self.DATA_FILE_WRITERS[format]
        filename, _ = os.path.splitext(str(path))  # Get rid of the extension (if any)
        is_binary = format in self.BINARY_FORMATS
        mode = "wb" if is_binary else "w"
        encoding = None if is_binary else "utf-8"
        with open(filename + "." + format, mode, encoding=encoding) as f:
            writer(self.data, f)

    def write_metadata(self, path: Union[str, Path]):
        filename, _ = os.path.splitext(str(path))  # Get rid of the extension (if any)
        with open(filename + "." + "metadata", "w", encoding="utf8") as f:
            json.dump(self.metadata, f)

    def write(self, basename: Union[str, Path], format: str):
        self.write_data(basename, format)
        if self.metadata is not None:
            self.write_metadata(basename)


####
# Main class
####
class LocalFileTree(Sequence):

    CONFIG_FILE: ClassVar[str] = ".filetree.json"
    LOCKFILE: ClassVar[str] = ".LOCK"
    DATA_DIR: ClassVar[str] = "data"

    def __init__(self, root: str = "./", readonly: bool = False, **kwargs) -> None:

        self._root = root
        self._readonly = readonly
        self._lock = fasteners.InterProcessReaderWriterLock(self._lock_file, logger=logger)
        self._last_config_read = datetime.datetime.min

        # Validate the root directory
        self._validate_root_dir()
        if not self._readonly:
            os.makedirs(os.path.join(self._root, self.DATA_DIR), exist_ok=True)

        with self._rw_lock_context():
            self._load_config(load_only_if_modified=False, **kwargs)

    @property
    def _config_file(self) -> str:
        return os.path.join(self._root, self.CONFIG_FILE)

    @property
    def _lock_file(self) -> str:
        return os.path.join(self._root, self.LOCKFILE)

    @property
    def _rw_lock_context(self) -> fasteners.ReaderWriterLock:
        return self._lock.read_lock if self._readonly else self._lock.write_lock

    def _validate_root_dir(self) -> None:
        if not os.path.exists(self._root):
            if self._readonly:
                raise InvalidRootDirException("Empy root directory in readonly mode")

        elif not os.path.isdir(self._root):
            raise InvalidRootDirException("%s exists but its not a directory" % self._root)

        elif len(os.listdir(self._root)) > 0:
            if not (os.path.exists(self._config_file) and os.path.isfile(self._config_file)):
                raise InvalidRootDirException("%s is not empty but the config file is missing" % self._root)

    def _load_config(self, load_only_if_modified: bool = False, **kwargs) -> LocalFileTreeConfig:

        if not os.path.exists(self._config_file):
            logger.info("New FileTree - Initializating config file")
            self._config = LocalFileTreeConfig(file_count=0, **kwargs)
            self._write_config()
            return self._config

        if load_only_if_modified:
            last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(self._config_file))
            last_read = self._last_config_read

            if last_modified <= last_read:
                logger.debug("Config file not been modified since the last read")
                return self._config
            else:
                logger.info("Config file modified (%s vs %s). Updating" % (last_read, last_modified))

        logger.debug("Reading Config file from %s" % self._config_file)
        self._config = LocalFileTreeConfig.parse_file(self._config_file)
        self._last_config_read = datetime.datetime.now()

        return self._config

    def _write_config(self) -> None:
        if self._readonly:
            raise InvalidTreeOperationException("Cannot modify the config file - Readonly mode")

        logger.info("Writing config file to %s" % self._config_file)
        with open(self._config_file, "w") as f:
            f.write(self._config.json())
        self._last_config_read = datetime.datetime.fromtimestamp(os.path.getmtime(self._config_file))

    def _decompose_index(self, value: int) -> List[int]:

        leaf_depth = self._config.leaf_depth
        depth = self._config.tree_depth
        bitmask = (1 << leaf_depth) - 1

        indexes = []
        for _ in range(depth):
            indexes.append(value & bitmask)
            value = value >> leaf_depth

        # Reverse the indexes so the low-depth
        # splits depend on leftmost bits.
        # If we add new files and need to add a new depth to the
        # tree, the existing tree is just a branch of the new one

        # Ignore the last split, its the position within the leaf.
        return indexes[1:][::-1]

    def _index_files(self, idx: int) -> Tuple[str, Optional[str]]:

        indexes = self._decompose_index(idx)
        indexes_str = [str(i) for i in indexes]
        base_name = os.path.join(self._root, self.DATA_DIR, *indexes_str)
        metadata_file: Optional[str] = None

        data_file = os.path.join(base_name, "%s.%s" % (str(idx), self._config.file_format))
        if self._config.has_metadata:
            metadata_file = os.path.join(base_name, "%s.metadata" % str(idx))
        else:
            metadata_file = None
        return data_file, metadata_file

    def get(self, idx: int) -> TreeDataEntry:

        with self._lock.read_lock():

            self._load_config(load_only_if_modified=True)

            if idx < 0:
                idx = self._config.file_count + idx

            if idx >= self._config.file_count:
                raise IndexError("Out of bounds")

            data, metadata = self._index_files(idx)
            entry = TreeDataEntry(data=data, metadata=metadata)
        return entry

    def _grow_tree(self, new_file_count: int) -> None:
        # Do we need to increase the tree's depth?
        max_bits = math.ceil(math.log2(new_file_count))
        required_depth = math.ceil(max_bits / self._config.leaf_depth) + 1
        current_depth = self._config.tree_depth

        logger.debug("Current Depth: %d, Required Depth: %d" % (current_depth, required_depth))
        if required_depth > self._config.tree_depth:

            extra_levels = required_depth - current_depth
            temporal_data_dir = os.path.join(self._root, ".GROWING." + self.DATA_DIR)
            logger.info(
                "Growing tree from depth %d to depth %d to accomodate %d entries"
                % (current_depth, required_depth, new_file_count)
            )

            # We need to create a new 0/0... branch and move
            # the existing tree there

            # Rename data to - data old and then create data/0/0/0..
            if self._config.file_count > 0:
                os.rename(os.path.join(self._root, self.DATA_DIR), temporal_data_dir)

                new_root_location = os.path.join(self._root, self.DATA_DIR, *["0" for i in range(extra_levels - 1)])
                os.makedirs(new_root_location, exist_ok=False)

                # Move data old to data/0/...
                os.rename(temporal_data_dir, os.path.join(new_root_location, "0"))

            # Update config
            self._config = LocalFileTreeConfig(tree_depth=required_depth, **self._config.dict(exclude={"tree_depth"}))
            self._write_config()

    def insert(self, values: Union[Sequence[TreeDataEntry], TreeDataEntry], progressbar: bool = True) -> Sequence[int]:

        if self._readonly:
            raise InvalidTreeOperationException("Insert Failed - Readonly mode")

        if isinstance(values, TreeDataEntry):
            values = [values]

        with self._lock.write_lock():

            # Update current config status if needed. No lock since
            # we already have an exclusive write lock
            self._load_config(load_only_if_modified=True)

            n_new_entries = len(values)
            prev_file_count = self._config.file_count
            new_file_count = prev_file_count + n_new_entries

            # Increase tree's depth if needed to acommodate new entires
            self._grow_tree(new_file_count)

            # At this point we know that we have enough capacity
            leaf_capacity = 2 ** self._config.leaf_depth
            progress_output = sys.stderr if progressbar else open(os.devnull, 'w')
            for idx, entry in tqdm.tqdm(
                enumerate(values, start=prev_file_count), file=progress_output, total=new_file_count
            ):

                data, metadata = self._index_files(idx)

                # We need to create the dir for the new leaf
                # That happens whenever idx mod leaf_capacity is 0
                if (idx % leaf_capacity) == 0:
                    logger.info("Creating new leaf at  %s" % (os.path.dirname(data)))
                    os.makedirs(os.path.dirname(data), exist_ok=True)

                entry.write_data(data, self._config.file_format)
                if self._config.has_metadata:
                    entry.write_metadata(metadata)

            # Update config with the new file count
            self._config = LocalFileTreeConfig(file_count=new_file_count, **self._config.dict(exclude={"file_count"}))
            self._write_config()

        # Return new file's indexes
        return range(prev_file_count, prev_file_count + n_new_entries)

    def __len__(self) -> int:
        return self._config.file_count

    def __getitem__(self, key: Union[int, slice]) -> Any:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            for i in range(start, stop, step):
                yield self.get(i)
        elif isinstance(key, int):
            return self.get(key)
        elif isinstance(key, tuple):
            raise NotImplementedError("Tuple as index")
        else:
            raise TypeError("Invalid argument type: {}".format(type(key)))
