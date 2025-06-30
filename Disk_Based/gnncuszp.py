import cuszp
import torch
from statcollector import StatCollector
# Create a class that performs compression and decompression on a tensor


class Compressor(torch.nn.Module):
    def __init__(self, err_mode, err_bound, device, num_nodes,statcollector:StatCollector):
        super(Compressor, self).__init__()
        self.err_mode = err_mode
        self.err_bound = err_bound
        self.device = device
        self.compressor = cuszp
        self.num_nodes = num_nodes
        self.sc = statcollector

    def compress(self, x):
        # Ensure float32 type
        if not x.dtype == torch.float32:
            raise TypeError("x must be of type torch.float32")
        x = x.contiguous()
        if self.err_mode == "rel" or self.err_mode == "relative":
            # Value-range error bound
            x_max = torch.max(x)
            x_min = torch.min(x)
            # Compute the err_bound
            err_bound = (x_max - x_min) * self.err_bound
            # print("min =", x_min, "max =", x_max, "err_bound =", err_bound)
            self.sc.add_tensor_stat("Min Value", x_min.item())
            self.sc.add_tensor_stat("Max Value", x_max.item())

        elif self.err_mode == "abs" or self.err_mode == "absolute":
            err_bound = self.err_bound
        else:
            raise ValueError("err_mode must be 'rel / relative' or 'abs / absolute'")
        self.sc.add_tensor_stat("Absolute Error Bound", err_bound.item())

        return CompressedElement(x, self.compressor.compress(x, err_bound, self.err_mode), err_bound, self.device)

    def decompress(self, comp_element):
        if not isinstance(comp_element, CompressedElement):
            raise TypeError("comp_element must be an instance of CompressedElement")
        compressed_size = (
            comp_element.compressed_data.numel()
            * comp_element.compressed_data.element_size()
        )
        decompressed = self.compressor.decompress(
            comp_element.compressed_data,
            comp_element.uncompressed_elements,
            compressed_size,
            comp_element.err_bound,
            self.err_mode,
        )
        # Reshape decompressed to match original shape
        decompressed = decompressed.reshape(comp_element.original_shape)
        return decompressed

    def pack_hook(self, x):
        if (
            x.dtype == torch.float32
            and not x.is_sparse
            and isinstance(x, torch.Tensor)
            and x.shape[0] >= self.num_nodes
        ):
            # print("Packing", x.shape)
            t0 = self.sc.new_clock()
            self.sc.sync_start_time(t0)

            compressed = self.compress(x)

            self.sc.sync_end_time(t0)
            self.sc.increment_epoch_stat("Total Compression Time (s)",self.sc.get_elapsed_time(t0))

            # print("Uncompressed size =", (x.numel() * x.element_size()) / 1024 / 1024)
            # print(
            #     "Compressed size =",
            #     (
            #         compressed.compressed_data.numel()
            #         * compressed.compressed_data.element_size()
            #     )
            #     / 1024
            #     / 1024,
            # )
            # print(
            #     "Compression Ratio = ",
            #     (x.numel() * x.element_size())
            #     / (
            #         compressed.compressed_data.numel()
            #         * compressed.compressed_data.element_size()
            #     ),
            # )
            csize = compressed.compressed_data.numel()*compressed.compressed_data.element_size()
            osize = x.numel() * x.element_size()
            self.sc.add_tensor_stat("Uncompressed Size (bytes)", osize)
            self.sc.add_tensor_stat("Compressed Size (bytes)", csize)
            self.sc.increment_epoch_stat("Average CR", osize/csize)
            self.sc.increment_epoch_stat("Aggregate Uncompressed Tensor Size (bytes)", osize)
            self.sc.increment_epoch_stat("Aggregate Compressed Tensor Size (bytes)", csize)
            # print( "Data Saved", ((x.numel() * x.element_size()) - (compressed.compressed_data.numel() * compressed.compressed_data.element_size()))/1024/1024)
            # print("Testing decompress,", decompressed)
            # print("Compressed data", compressed.compressed_data)
            # print("Decompressed shape =", decompressed.shape)
            # print("X shape = ", x.shape)
            # abs_error = torch.abs(x - decompressed)
            # max_error = torch.max(abs_error)
            # if max_error > self.err_bound * 1.1:
            #     # Print the location of the max error and the values
            #     print("Max error location =", torch.argmax(torch.abs(x - decompressed)))
            #     print("Max error value =", max_error)
            #     location = torch.argmax(torch.abs(x - decompressed))
            #     # Print row and column of max error
            #     print("Row =", int(location / x.shape[1]))
            #     print("Column =", location % x.shape[1])
            #     # Count the number of elements that are > self.err_bound * 1.1
            #     bound_err_cnt = torch.sum(abs_error > self.err_bound * 1.1)
            #     print("Number of elements > err_bound * 1.1 =", bound_err_cnt)
            #     print("X value =", x[int(location / x.shape[1])][location % x.shape[1]])
            #     print(
            #         "Decompressed value =",
            #         decompressed[int(location / x.shape[1])][location % x.shape[1]],
            #     )
            #     raise ValueError(
            #         "Error bound exceeded! Max error = ", max_error
            #     )
            # # Ensure max_error <= err_bound

            # print("Max error =", max_error)
            # Ensure x is freed
            # delete x
            self.sc.increment_epoch_stat("Compressed Tensor Count",1)
            self.sc.register_tensor_row_and_update()


            del x
            # empty cache
            torch.cuda.empty_cache()
            return compressed
        else:
            return x

    def unpack_hook(self, x):
        if isinstance(x, CompressedElement):
            # print("Unpacking", x.name)
            # print("Unpacking")
            t0 = self.sc.new_clock()
            self.sc.sync_start_time(t0)

            decompressed = self.decompress(x)

            self.sc.sync_end_time(t0)
            self.sc.increment_epoch_stat("Total Decompression Time (s)",self.sc.get_elapsed_time(t0))

            # print("Unpacked")
            # print("Unpacked to", decompressed)
            return decompressed
        else:
            return x
    
    def compress_async(self, x):
        # Ensure float32 type
        if not x.dtype == torch.float32:
            raise TypeError("x must be of type torch.float32")
        x = x.contiguous()
        if self.err_mode == "rel" or self.err_mode == "relative":
            # Value-range error bound
            x_max = torch.max(x)
            x_min = torch.min(x)
            # Compute the err_bound
            err_bound = (x_max - x_min) * self.err_bound
            # print("min =", x_min, "max =", x_max, "err_bound =", err_bound)
            self.sc.add_tensor_stat("Min Value", x_min.item())
            self.sc.add_tensor_stat("Max Value", x_max.item())

        elif self.err_mode == "abs" or self.err_mode == "absolute":
            err_bound = self.err_bound
        else:
            raise ValueError("err_mode must be 'rel / relative' or 'abs / absolute'")
        self.sc.add_tensor_stat("Absolute Error Bound", err_bound.item())


        compressed = torch.zeros((x.numel() * x.element_size()), dtype=torch.uint8, device=self.device)
        csize = self.compressor.compress_async(x, compressed, err_bound, self.err_mode)
        return CompressedElement(x, compressed.resize_(csize), err_bound, self.device)

    def async_pack_hook(self, x):
        if (
            x.dtype == torch.float32
            and not x.is_sparse
            and isinstance(x, torch.Tensor)
            and x.shape[0] >= self.num_nodes
        ):

            s0 = torch.cuda.Stream()
            s0.wait_stream(torch.cuda.default_stream(self.device))
            with torch.cuda.stream(s0):
                x.record_stream(s0)
                osize = x.numel() * x.element_size()
                # Create device tensor the same size as x, but as uint8
                compressed = self.compress_async(x)
                # Resize the tensor to csize
                # Free x
                del x
                # csize = compressed.compressed_data.numel()*compressed.compressed_data.element_size()
                # self.sc.add_tensor_stat("Uncompressed Size (bytes)", osize)
                # self.sc.add_tensor_stat("Compressed Size (bytes)", csize)
                # self.sc.increment_epoch_stat("Average CR", osize/csize)
                # self.sc.increment_epoch_stat("Aggregate Uncompressed Tensor Size (bytes)", osize)
                # self.sc.increment_epoch_stat("Aggregate Compressed Tensor Size (bytes)", csize)
                self.sc.increment_epoch_stat("Compressed Tensor Count",1)
                self.sc.register_tensor_row_and_update()
            return (s0, compressed)
        else:
            return (None, x)

    def async_unpack_hook(self, x):
        if isinstance(x[1], CompressedElement):
            # print("Unpacking", x.name)
            # print("Unpacking")
            s0, compressed = x
            s0.wait_stream(s0)
            # t0 = self.sc.new_clock()
            # self.sc.sync_start_time(t0)

            decompressed = self.decompress(compressed)

            # self.sc.sync_end_time(t0)
            # self.sc.increment_epoch_stat("Total Decompression Time (s)",self.sc.get_elapsed_time(t0))

            # print("Unpacked")
            # print("Unpacked to", decompressed)
            return decompressed
        else:
            return x[1]


# Create class for a compressed element that is used by the Compressor class


class CompressedElement(torch.nn.Module):
    def __init__(self, x, compressed, err_bound, device):
        super(CompressedElement, self).__init__()
        self.device = device
        # self.compressor = cuszp
        self.compressed_data = compressed
        self.uncompressed_elements = x.numel()
        self.original_shape = x.shape
        self.err_bound = err_bound