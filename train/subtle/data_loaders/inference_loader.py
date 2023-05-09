import numpy as np
import sigpy as sp

from torch.utils.data import Dataset
from typing import Tuple

class InferenceLoader(Dataset):
    """
    Sequence data loader for training 2.5D models. 2.5D is a method by which a few adjacent slices
    are appended and passed as input to the network to predict the center slice. The class is
    initialized with a 4D numpy array of the input data of shape (C x N x H x W) where C is the
    number of channels of different image sets that a project uses - Ex: this is typically 2 for
    SubtleGad as the image sets are zero dose and low dose images for contrast reduction. N is the
    number of slices and H & W, height and width respectively.

    The class is also initialized with `slices_per_input` which specifies the number of adjacent
    slices to be taken for context and the `slice_axis` gives the flexibility to slice the input
    volume in any given orientation which is useful for multi-planar reformatting (MPR). When
    `slices_per_input = 1`, this class can be used as a 2D data loader.
    """
    def __init__(self, input_data: np.ndarray, slices_per_input: int, batch_size: int,
                 slice_axis: int = 1, resize: Tuple[int, int] = None,
                 data_order: str = 'stack'):
        """
        Initialize the data loader class with input_data and other arguments

        :param input_data: The numpy array of shape (C, N, H, W) where C is the number of image
        sets, N - number of slices, H & W - height and width of the image respectively
        :param slices_per_input: Number of adjacent slices for 2.5D context
        :param batch_size: Length of each batch of data
        :param slice_axis: Specifies the axis to be used as the slice axis to support different
        orientations
        :param resize: (height, width) - shape to which the generated slices are to be resized
        :param data_order: specifies how the image sets need to be arranged
        - 'interleave': alternate between the different image set - Ex: Arrange the data as (zero,
        low, zero, low ...) for SubtleGAD
        - 'stack': stack the image sets one followed by the other
        """
        assert input_data.ndim == 4, "input_data should be a 4D array"
        assert slices_per_input >= 1, "slices_per_input should be >= 1"
        assert slice_axis in [0, 2, 3], "Invalid slices_axis - must be 0, 2 or 3"
        assert data_order in ['interleave', 'stack'], "data_order must be 'interleave' or 'stack'"

        self.slices_per_input = slices_per_input
        self.batch_size = batch_size
        self.resize = resize
        self.data_order = data_order

        if resize is not None:
            assert_str = "resize must be a tuple of length two - (height, width)"
            assert (isinstance(resize, tuple) and len(resize) == 2), assert_str

        if slice_axis == 0:
            self.input_data = np.copy(input_data)
        elif slice_axis == 2:
            self.input_data = np.copy(input_data).transpose(0, 2, 1, 3)
        elif slice_axis == 3:
            self.input_data = np.copy(input_data).transpose(0, 3, 1, 2)

        self.num_slices = self.input_data.shape[1]

    def __len__(self) -> int:
        """
        Mandatory method required by keras.utils.Sequence. Returns the total number of batches for
        a given data volume
        """
        batches = int(np.floor(self.num_slices / self.batch_size))
        if batches * self.batch_size != self.num_slices:
            batches += 1
        return batches

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Mandatory method required by keras.utils.Sequence. Gets the batch data for a given position
        `index`

        :param index: Position of the batch
        :return: Array of shape (batch_size, H, W, num_img_sets * slices_per_input)
        """
        start_idx = (index * self.batch_size)
        end_idx = ((index + 1) * self.batch_size)
        batch_slices = []

        for idx in range(start_idx, end_idx):
            cwidth = self.slices_per_input // 2

            # 2.5D
            slice_idxs = np.arange(idx - cwidth, idx + cwidth + 1)
            # handle edge cases
            slice_idxs = np.minimum(np.maximum(slice_idxs, 0), self.num_slices - 1)
            slices = self.input_data[:, slice_idxs, ...].transpose(1, 0, 2, 3)[None, ...]
            if self.resize is not None:
                slices = sp.util.resize(slices, (
                    slices.shape[0], slices.shape[1], slices.shape[2],
                    self.resize[0], self.resize[1]
                ))

            batch_slices.append(slices)

        X = (
            np.array(batch_slices[0])
            if len(batch_slices) == 1
            else np.concatenate(batch_slices, axis=0)
        )
        # pylint: disable=too-many-function-args
        X = np.reshape(X, (X.shape[0], -1, X.shape[3], X.shape[4]), order='F')
        return X
