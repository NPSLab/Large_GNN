import pandas as pd
import numpy as np
import time
import torch

EPOCH_STATISTICS = ["Epoch",
              "Training Time",
              "Testing Time",
              "Peak Memory Consumption",
              "Forward Pass Memory Allocated (bytes)",
              "Training Loss",
              "Training Accuracy",
              "Validation Accuracy",
              "Test Accuracy"]

CPU_STATISTICS = ["Total D2H Time (s)",
                  "Total H2D Time (s)",
                  "Transferred Tensor Count",
                  "Aggregate Tensor Size (bytes)"]

EPOCH_COMPRESSION_STATISTICS = ["Average CR",
                                "Total Compression Time (s)",
                                "Total Decompression Time (s)",
                                "Compressed Tensor Count",
                                "Aggregate Uncompressed Tensor Size (bytes)",
                                "Aggregate Compressed Tensor Size (bytes)"]

TENSOR_COMPRESSION_STATISTICS = ["Tensor",
                                 "Min Value",
                                 "Max Value",
                                 "Absolute Error Bound",
                                 "Uncompressed Size (bytes)",
                                 "Compressed Size (bytes)"]

class StatCollector():
    def __init__(self, model_type: str):
        self.iscpu = (model_type =="cpu")
        self.iscompressed = (model_type =="compressed")
        self.isbase = (model_type =="base")

        assert(self.iscpu or self.iscompressed or self.isbase)

        self.epochstats = EPOCH_STATISTICS
        self.current_epoch = 0
        if self.iscpu:
            self.epochstats += CPU_STATISTICS
        elif self.iscompressed:
            self.epochstats += EPOCH_COMPRESSION_STATISTICS
            self.tensorstats = TENSOR_COMPRESSION_STATISTICS

            self.tensor_entry = {key: 0.0 for key in self.tensorstats}
            self.current_tensor = 0
            self.tensor_entry["Tensor"] = self.current_tensor
            self.tensor_entries = []

        self.epoch_entry = {key: 0.0 for key in self.epochstats}

        self.epoch_entries = []
        
        self.epoch_entry["Epoch"] = self.current_epoch

        self.clock_count = 0
        self.s_time = []
        self.e_time = []

    def reset_stats(self):
        self.epochstats = EPOCH_STATISTICS
        self.current_epoch = 0
        if self.iscpu:
            self.epochstats += CPU_STATISTICS
        elif self.iscompressed:
            self.epochstats += EPOCH_COMPRESSION_STATISTICS
            self.tensorstats = TENSOR_COMPRESSION_STATISTICS

            self.tensor_entry = {key: 0.0 for key in self.tensorstats}
            self.current_tensor = 0
            self.tensor_entry["Tensor"] = self.current_tensor
            self.tensor_entries = []

        self.epoch_entry = {key: 0.0 for key in self.epochstats}

        self.epoch_entries = []
        
        self.epoch_entry["Epoch"] = self.current_epoch

        self.clock_count = 0
        self.s_time = []
        self.e_time = []

    def save_dfs(self, epoch_file_name="epochstats.out", tensor_file_name="tensorstats.out"):
        
        if self.iscompressed:
            tensordf = pd.DataFrame(self.tensor_entries)
            tensordf.to_csv(tensor_file_name)
        epochdf = pd.DataFrame(self.epoch_entries)
        epochdf.to_csv(epoch_file_name)

    def register_epoch_row_and_update(self):
        self.epoch_entries.append(self.epoch_entry)
        self.epoch_entry = {key: 0.0 for key in self.epochstats}
        self.current_epoch += 1
        self.epoch_entry["Epoch"] = self.current_epoch

    def register_tensor_row_and_update(self):
        self.tensor_entries.append(self.tensor_entry)
        self.tensor_entry = {key: 0.0 for key in self.tensorstats}
        self.current_tensor += 1
        self.tensor_entry["Tensor"] = self.current_tensor

    def new_clock(self):
        self.s_time.append(0.0)
        self.e_time.append(0.0)
        self.clock_count+=1
        return self.clock_count-1

    def clear_clocks(self):
        self.clock_count = 0
        self.s_time = []
        self.e_time = []

    def sync_start_time(self, clock: int):
        torch.cuda.synchronize()
        self.s_time[clock] = time.time()

    def sync_end_time(self, clock: int):
        torch.cuda.synchronize()
        self.e_time[clock] = time.time()

    def nosync_start_time(self, clock: int):
        self.s_time[clock] = time.time()

    def nosync_end_time(self, clock: int):
        self.e_time[clock] = time.time()


    def get_elapsed_time(self, clock: int):
        return self.e_time[clock] - self.s_time[clock]
    
    def add_epoch_stat(self, key : str, value):
        self.epoch_entry[key] = value

    def add_tensor_stat(self, key : str, value):
        self.tensor_entry[key] = value

    def increment_epoch_stat(self, key : str, value):
        self.epoch_entry[key] += value
    
    def increment_tensor_stat(self, key : str, value):
        self.tensor_entry[key] += value