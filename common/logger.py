import os
import sys
import json
import pprint
import argparse
import datetime
import warnings
import numpy as np

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
from tokenize import Number
from torch.utils.tensorboard import SummaryWriter


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
BACKUP = 60

DEFAULT_X_NAME = "timestep"
ROOT_DIR = "log"


class KVWriter(object):
    """
    Key Value writer
    """
    def writekvs(self, kvs: Dict) -> None:
        """
        write a dictionary to file
        """
        raise NotImplementedError


class StrWriter(object):
    """
    string writer
    """
    def writestr(self, s: str) -> None:
        """
        write a string to file
        """
        raise NotImplementedError


class StandardOutputHandler(KVWriter, StrWriter):
    def __init__(self, filename_or_textio: Union[str, TextIO]) -> None:
        """
        log to a file, in a human readable format

        :param filename_or_file: (str or File) the file to write the log to
        """
        if isinstance(filename_or_textio, str):
            self.file = open(filename_or_textio+".txt", 'at')
            self.own_file = True
            self.handler_name = os.path.basename(filename_or_textio)
        else:
            assert hasattr(filename_or_textio, 'write'), 'Expected file or str, got {}'.format(filename_or_textio)
            self.file = filename_or_textio
            self.own_file = False
            self.handler_name = "stdio"
        super().__init__()

    def writekvs(self, kvs: Dict) -> None:
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn('Tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 40)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s: str) -> str:
        return s[:40] + '...' if len(s) > 80 else s

    def writestr(self, s: str) -> None:
        self.file.write(s)
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


class JSONOutputHandler(KVWriter):
    def __init__(self, filename: str) -> None:
        """
        log to a file in the JSON format
        """
        self.file = open(filename+".json", 'at')
        self.handler_name = os.path.basename(filename)
        super().__init__()

    def writekvs(self, kvs: Dict) -> None:
        for key, value in sorted(kvs.items()):
            if hasattr(value, 'dtype'):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    kvs[key] = float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    kvs[key] = value.tolist()
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class CSVOutputHandler(KVWriter):
    def __init__(self, filename: str) -> None:
        """
        log to a file in the CSV format
        """
        filename += ".csv"
        self.filename = filename
        self.file = open(filename, 'a+t')
        self.file.seek(0)
        keys = self.file.readline()
        if keys != '':
            keys = keys[:-1] # skip '\n'
            keys = keys.split(',')
            self.keys = keys
        else:
            self.keys = []
        self.file = open(filename, 'a+t')
        self.sep = ','
        self.handler_name = os.path.splitext(os.path.basename(filename))[0]
        super().__init__()

    def writekvs(self, kvs: Dict) -> None:
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file = open(self.filename, 'w+t')
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(key)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
            self.file = open(self.filename, 'a+t')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = kvs.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class TensorBoardOutputHandler(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, filename: str) -> None:
        self.step = 1
        self.tb_writer = SummaryWriter(filename)
        self.handler_name = os.path.basename(filename)
        super().__init__()

    @property
    def writer(self) -> SummaryWriter:
        return self.tb_writer

    def add_hyper_params_to_tb(self, hyper_param: Dict, metric_dict=None) -> None:
        if metric_dict is None:
            pp = pprint.PrettyPrinter(indent=4)
            self.writer.add_text('hyperparameters', pp.pformat(hyper_param))
        else:
            self.writer.add_hparams(hyper_param, metric_dict)

    def writekvs(self, kvs: Dict) -> None:
        def summary_val(k, v):
            kwargs = {'tag': k, 'scalar_value': float(v), 'global_step': self.step}
            self.writer.add_scalar(**kwargs)

        for k, v in kvs.items():
            if k == DEFAULT_X_NAME: continue
            summary_val(k, v)
    
    def set_step(self, step: int) -> None:
        self.step = step

    def close(self) -> None:
        if self.writer:
            self.writer.close()


HANDLER = {
    "stdout": StandardOutputHandler,
    "csv": CSVOutputHandler,
    "tensorboard": TensorBoardOutputHandler
}


class Logger(object):
    def __init__(self, dir: str, ouput_config: Dict) -> None:
        self._dir = dir
        self._init_dirs()
        self._init_ouput_handlers(ouput_config)
        self._name2val = defaultdict(float)
        self._name2cnt = defaultdict(int)
        self._level = INFO
        self._timestep = 0
    
    def _init_dirs(self) -> None:
        self._record_dir = os.path.join(self._dir, "record")
        self._checkpoint_dir = os.path.join(self._dir, "checkpoint")
        self._model_dir = os.path.join(self._dir, "model")
        self._result_dir = os.path.join(self._dir, "result")
        os.mkdir(self._record_dir)
        os.mkdir(self._checkpoint_dir)
        os.mkdir(self._model_dir)
        os.mkdir(self._result_dir)
    
    def _init_ouput_handlers(self, output_config: Dict) -> None:
        self._output_handlers = []
        for file_name, fmt in output_config.items():
            try:
                self._output_handlers.append(HANDLER[fmt](os.path.join(self._record_dir, file_name)))
            except KeyError:
                warnings.warn("Invalid output type, Valid types: stdout, csv, tensorboard", DeprecationWarning)
        # default output to console
        self._output_handlers.append(StandardOutputHandler(sys.stdout))
    
    def log_hyperparameters(self, hyper_param: Dict) -> None:
        json_output_handler = JSONOutputHandler(os.path.join(self._record_dir, "hyper_param"))
        json_output_handler.writekvs(hyper_param)
        json_output_handler.close()
        for handler in self._output_handlers:
            if isinstance(handler, TensorBoardOutputHandler):
                handler.add_hyper_params_to_tb(hyper_param)

    def logkv(self, key: Any, val: Any) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.
        """
        self._name2val[key] = val

    def logkv_mean(self, key: Any, val: Number) -> None:
        """
        The same as logkv(), but if called many times, values averaged.
        """
        oldval, cnt = self._name2val[key], self._name2cnt[key]
        self._name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self._name2cnt[key] = cnt + 1

    def dumpkvs(self, exclude:Optional[Union[str, Tuple[str, ...]]]=None) -> None:
        # log timestep
        self.logkv(DEFAULT_X_NAME, self._timestep)
        for handler in self._output_handlers:
            if isinstance(handler, KVWriter):
                if exclude is not None and handler.handler_name in exclude:
                    continue
                handler.writekvs(self._name2val)
        self._name2val.clear()
        self._name2cnt.clear()

    def log(self, s: str, level=INFO) -> None:
        for handler in self._output_handlers:
            if isinstance(handler, StandardOutputHandler):
                handler.writestr(s)
    
    def set_timestep(self, timestep: int) -> None:
        self._timestep = timestep
        for handler in self._output_handlers:
            if isinstance(handler, TensorBoardOutputHandler):
                handler.set_step(timestep)

    def set_level(self, level) -> None:
        self._level = level

    @property
    def record_dir(self) -> str:
        return self._record_dir
    
    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir

    @property
    def model_dir(self) -> str:
        return self._model_dir
    
    @property
    def result_dir(self) -> str:
        return self._result_dir
    
    def close(self) -> None:
        for handler in self._output_handlers:
            handler.close()


def make_log_dirs(
    task_name: str,
    algo_name: str,
    seed: int,
    args: Dict,
    record_params: Optional[List]=None
) -> str:
    if record_params is not None:
        for param_name in record_params:
            algo_name += f"&{param_name}={args[param_name]}"
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"seed_{seed}&timestamp_{timestamp}"
    log_dirs = os.path.join(ROOT_DIR, task_name, algo_name, exp_name)
    os.makedirs(log_dirs)
    return log_dirs


def load_args(load_path: str) -> argparse.ArgumentParser:
    args_dict = {}
    with open(load_path,'r') as f:
        args_dict.update(json.load(f))
    return argparse.Namespace(**args_dict)
