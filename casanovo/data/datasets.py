"""A PyTorch Dataset class for annotated spectra."""

from typing import Optional, Tuple

from ..depthcharge.data.hdf5 import SpectrumIndex
import numpy as np
import spectrum_utils.spectrum as sus
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    """
    Parse and retrieve collections of MS/MS spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None`
        retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude
        TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the
        base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the
        precursor mass.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        spectrum_index: SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index

    def __len__(self) -> int:
        """The number of spectra."""
        return self.n_spectra

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, float, int, Tuple[str, str]]:
        """
        Return the MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        spectrum_id: Tuple[str, str]
            The unique spectrum identifier, formed by its original peak file and
            identifier (index or scan number) therein.
        """
        mz_array, int_array, precursor_mz, precursor_charge = self.index[idx]
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )
        return (
            spectrum,
            precursor_mz,
            precursor_charge,
            self.get_spectrum_id(idx),
        )

    def get_spectrum_id(self, idx: int) -> Tuple[str, str]:
        """
        Return the identifier of the MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the MS/MS spectrum within the SpectrumIndex.

        Returns
        -------
        ms_data_file : str
            The peak file from which the MS/MS spectrum was originally parsed.
        identifier : str
            The MS/MS spectrum identifier, per PSI recommendations.
        """
        with self.index:
            return self.index.get_spectrum_id(idx)

    def _process_peaks(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.Tensor:
        """
        Preprocess the spectrum by removing noise peaks and scaling the peak
        intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak m/z values.
        int_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak intensity values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        """
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz_array.astype(np.float64),
            int_array.astype(np.float32),
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.scale_intensity("root", 1)
            intensities = spectrum.intensity / np.linalg.norm(
                spectrum.intensity
            )
            return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
        except ValueError:
            # Replace invalid spectra by a dummy spectrum.
            return torch.tensor([[0, 1]]).float()

    def _process_peaks_optimized(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.Tensor:
        """
        Optimized preprocessing of spectrum peaks for better performance.

        This version replaces spectrum_utils operations with direct numpy
        operations for significant speed improvements.
        """
        try:
            # Convert to numpy arrays if needed
            if not isinstance(mz_array, np.ndarray):
                mz_array = np.array(mz_array, dtype=np.float64)
            if not isinstance(int_array, np.ndarray):
                int_array = np.array(int_array, dtype=np.float32)

            # Early return for empty spectra
            if len(mz_array) == 0 or len(int_array) == 0:
                return torch.tensor([[0, 1]], dtype=torch.float32)

            # 1. Filter by m/z range (vectorized)
            mz_mask = (mz_array >= self.min_mz) & (mz_array <= self.max_mz)
            if not np.any(mz_mask):
                return torch.tensor([[0, 1]], dtype=torch.float32)

            mz_filtered = mz_array[mz_mask]
            int_filtered = int_array[mz_mask]

            # 2. Remove precursor peak (vectorized)
            precursor_mass_tolerance = self.remove_precursor_tol
            precursor_mask = np.abs(mz_filtered - precursor_mz) > precursor_mass_tolerance
            if not np.any(precursor_mask):
                return torch.tensor([[0, 1]], dtype=torch.float32)

            mz_no_precursor = mz_filtered[precursor_mask]
            int_no_precursor = int_filtered[precursor_mask]

            # CRITICAL: Sort by m/z like spectrum_utils does in constructor
            # This ensures the same ordering behavior as spectrum_utils
            mz_order = np.argsort(mz_no_precursor)
            mz_no_precursor = mz_no_precursor[mz_order]
            int_no_precursor = int_no_precursor[mz_order]

            # 3. Apply filter_intensity logic exactly like spectrum_utils
            # This matches spectrum_utils.filter_intensity(min_intensity, max_num_peaks)
            max_num_peaks = self.n_peaks if self.n_peaks is not None else len(int_no_precursor)
            
            # Sort ALL peaks by intensity (like spectrum_utils does)
            intensity_idx = np.argsort(int_no_precursor)
            min_intensity_threshold = self.min_intensity * int_no_precursor[intensity_idx[-1]]
            
            # Find start index (first peak above threshold)
            start_i = 0
            for intens in int_no_precursor[intensity_idx]:
                if intens > min_intensity_threshold:
                    break
                start_i += 1
            
            # Create mask using spectrum_utils exact logic
            mask = np.full_like(int_no_precursor, False, dtype=bool)
            mask[intensity_idx[max(start_i, len(intensity_idx) - max_num_peaks):]] = True
            
            mz_final = mz_no_precursor[mask]
            int_final = int_no_precursor[mask]
            
            # Early return if no peaks survive
            if len(mz_final) == 0:
                return torch.tensor([[0, 1]], dtype=torch.float32)

            # 4. Scale intensity (square root + normalization)
            # Match spectrum_utils exactly: use power function and same data types
            int_scaled = np.power(int_final, 1.0/2.0)
            intensities = int_scaled / np.linalg.norm(int_scaled)

            # 5. Create tensor exactly like the original code
            return torch.tensor(np.array([mz_final, intensities])).T.float()

        except (ValueError, IndexError, ZeroDivisionError):
            # Return dummy spectrum for invalid cases
            return torch.tensor([[0, 1]], dtype=torch.float32)

    @property
    def n_spectra(self) -> int:
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self) -> SpectrumIndex:
        """The underlying SpectrumIndex."""
        return self._index

    @property
    def rng(self):
        """The NumPy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the NumPy random number generator."""
        self._rng = np.random.default_rng(seed)


class AnnotatedSpectrumDataset(SpectrumDataset):
    """
    Parse and retrieve collections of annotated MS/MS spectra.

    Parameters
    ----------
    annotated_spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None`
        retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude
        TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the
        base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the
        precursor mass.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        annotated_spectrum_index:SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
        use_optimized_processing: bool = True,
    ):
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            remove_precursor_tol=remove_precursor_tol,
            random_state=random_state,
        )
        self.use_optimized_processing = use_optimized_processing

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int, str]:
        """
        Return the annotated MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        annotation : str
            The peptide annotation of the spectrum.
        """
        (
            mz_array,
            int_array,
            precursor_mz,
            precursor_charge,
            peptide,
        ) = self.index[idx]
        
        # Choose processing method based on optimization setting
        if self.use_optimized_processing:
            spectrum = self._process_peaks_optimized(
                mz_array, int_array, precursor_mz, precursor_charge
            )
        else:
            spectrum = self._process_peaks(
                mz_array, int_array, precursor_mz, precursor_charge
            )
        
        return spectrum, precursor_mz, precursor_charge, peptide
