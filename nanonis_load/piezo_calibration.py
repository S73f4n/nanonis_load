import matplotlib.backend_bases
import matplotlib.pyplot as plt
import numpy as np

from . import sxm, util


def transform_coordinates(x, y, transformation_matrix) -> np.ndarray:
    """
    Returns x and y coordinates after being transformed by transformation_matrix.
    x and y should be 2D arrays of the same shape.

    Parameters
    ----------
    x : NDArray
        The x coordinates (should come from np.meshgrid())
    y : NDArray
        The y coordinates (should come from np.meshgrid())
    """
    XY_combined = np.stack([x, y])
    X_transformed, Y_transformed = np.einsum(
        "ij, jkl", transformation_matrix, XY_combined
    )
    return X_transformed, Y_transformed  # Returns [x, y]


class PiezoCalibrationGraphene:
    """
    Determines the piezo calibration transformation matrix using an
    atomic scale image of graphene. The constructor takes the path to a .sxm file
    as its main argument and will open a matplotlib window with four subplots.
    The top left will have the real space image, the top right will have the FFT
    of the real space image, and the bottom two will be empty. Peaks in the FFT
    will be found by clicking near a peak. A 2D Gaussian fit will be performed on
    the window shown by the blue square following the mouse and the peak position
    will be extracted. You must select two peaks that you believe should be
    60 degrees apart and of the same magnitude. Once two peaks have been selected,
    a transformation matrix will be computed and applied to the FFT image (and its
    inverse transpose will be applied to the real space image) and the result will
    be plotted in the bottom two panels. You can use this to verify if the image
    is transformed properly.
    """

    def __init__(
        self,
        file: str,
        channel: str = "Z (m)",
        direction=0,
        linear_by_line: bool = False,
        rasterized: bool = True,
        window_function: str = "rectangular",
        min_amplitude=0.1,
        max_filter_width=5,
        fit_radius=5,
        cmap="Greys_r",
        processing=None,
        align_x_axis=False,
    ):
        """
        Parameters
        ----------
        file_path : str
            The path of the .sxm file to be opened
        channel : str, optional
            The channel of the scan file to be opened. Defaults to 'Z (m)'
        direction : bool, optional
            The scan direction to open. 0 = forward, 1 = backward.
        auto_fft_colormap : bool, optional
            If true, the vmax and vmin will be 3*sigma away from the mean for
            the FFT.
        linear_by_line : bool, optional
            If true, a linear background subtraction for each row of the data
            is done.
        rasterized : bool, optional
            Rasterizes the images.
        colorbar : bool, optional
            Whether or not to put a colorbar on each panel.
        image_resolution : int, optional
            The image will be downsampled to have this resolution (along with
            the FFT). This does NOT downsample the actual data. It only decreases
            the resolution of the image to make the program run more smoothly.
            Fitting and peak finding are done on the full data.
        window_function : str, optional
            The type of window function to use for the FFT. Default is 'rectangular'.
            Other options include 'blackman'.
        fft_mode : str, optional
            Which part of the FFT to plot. Default is 'amplitude' and other options
            include 'phase', 'real', and 'imag'.

        """

        self.cmap = cmap

        fig, ax = plt.subplots(2, 2)
        self.fig = fig

        self.real_ax = ax[0][0]
        self.fft_ax = ax[0][1]
        self.transformed_real_ax = ax[1][0]
        self.transformed_fft_ax = ax[1][1]
        self.align_x_axis = align_x_axis

        if isinstance(file, str):
            self.sxm_data = sxm.Sxm(file)
        elif isinstance(file, sxm.Sxm):
            self.sxm_data = file
        else:
            raise TypeError("'file' should be a str or sxm")

        # Plot image data
        if linear_by_line:
            self.image_data = self.sxm_data.subtract_linear_by_line(channel, direction)
        else:
            self.image_data = self.sxm_data.subtract_plane(channel, direction)

        if processing is not None:
            for process in processing:
                self.image_data = sxm.Sxm.process_data(self.image_data, process)

        self.real_im_plot = self.real_ax.imshow(
            self.image_data,
            origin="lower",
            cmap=cmap,
            extent=(0, self.sxm_data.x_range, 0, self.sxm_data.y_range),
        )

        # Plot fft
        self.fft_image_data = np.abs(
            util.correct_fft2D(self.image_data, window_function)
        )
        fft_x_range = self.sxm_data.x_pixels / self.sxm_data.x_range
        fft_y_range = self.sxm_data.y_pixels / self.sxm_data.y_range
        self.fft_clim = (0, np.std(self.fft_image_data))
        self.fft_im_plot = self.fft_ax.imshow(
            self.fft_image_data,
            origin="lower",
            cmap=cmap,
            extent=(
                -fft_x_range / 2,
                fft_x_range / 2,
                -fft_y_range / 2,
                fft_y_range / 2,
            ),
            clim=self.fft_clim,
            rasterized=True,
        )

        kx = np.linspace(-fft_x_range / 2, fft_x_range / 2, self.sxm_data.x_pixels)
        ky = np.linspace(-fft_y_range / 2, fft_y_range / 2, self.sxm_data.y_pixels)
        KX, KY = np.meshgrid(kx, ky)
        self.fft_peaks = util.find_peaks2D(
            KX,
            KY,
            self.fft_image_data,
            min_amplitude=min_amplitude,
            max_filter_width=max_filter_width,
            fit_radius=fit_radius,
        )

        # Find fft peaks and plot them
        self.fft_peak_scatter = self.fft_ax.scatter(
            self.fft_peaks[:, 0],
            self.fft_peaks[:, 1],
            color=[[0.12156863, 0.46666667, 0.70588235, 1.0]] * len(self.fft_peaks),
        )
        self.fft_ax.set_xlim(-fft_x_range / 2, fft_x_range / 2)
        self.fft_ax.set_ylim(-fft_y_range / 2, fft_y_range / 2)
        self.fft_peak_scatter_shown = True
        self.clicked_peaks = []

        self.transformed_real_ax.set_aspect("equal")
        self.transformed_fft_ax.set_aspect("equal")

        self.clear_transformed_plots()

        # Event binding
        self.button_press_bind = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_button_press
        )

        plt.show()

        plt.get_current_fig_manager().window.showMaximized()

    def set_real_cmap(self, cmap):
        self.real_im_plot.set_cmap(cmap)
        if self.transformed_im_plot is not None:
            self.transformed_im_plot.set_cmap(cmap)
        self.fig.canvas.draw()

    def set_im_clim(self, vmin, vmax):
        self.real_im_plot.set_clim(vmin, vmax)
        if self.transformed_im_plot is not None:
            self.transformed_im_plot.set_clim(vmin, vmax)
        self.fig.canvas.draw()

    def set_fft_cmap(self, cmap):
        self.fft_im_plot.set_cmap(cmap)
        if self.transformed_fft_plot is not None:
            self.transformed_fft_plot.set_cmap(cmap)
        self.fig.canvas.draw()

    def set_fft_clim(self, vmin, vmax):
        self.fft_im_plot.set_clim(vmin, vmax)
        if self.transformed_fft_plot is not None:
            self.transformed_fft_plot.set_clim(vmin, vmax)
        self.fig.canvas.draw()

    def hide_peak_scatter(self):
        if self.fft_peak_scatter_shown:
            self.old_peak_colors = self.fft_peak_scatter.get_facecolor()
            self.fft_peak_scatter.set_facecolor((0, 0, 0, 0))
            self.fft_peak_scatter.set_edgecolor((0, 0, 0, 0))
            self.fig.canvas.draw()
            self.fft_peak_scatter_shown = False

    def show_peak_scatter(self):
        if not self.fft_peak_scatter_shown:
            self.fft_peak_scatter.set_facecolor(self.old_peak_colors)
            self.fft_peak_scatter.set_edgecolor(self.old_peak_colors)
            self.fig.canvas.draw()
            self.fft_peak_scatter_shown = True

    def get_fft_transformation_matrix(
        self,
    ):  # Returns the real space transformation matrix determined by two selected FFT peaks (it does an inverse transpose at the end).
        """
        Returns the FFT transformation matrix computed using the two selected
        points. This matrix will only properly transform the image in reciprocal
        space. The corresponding matrix in real space is its inverse transpose,
        which you can get using get_real_space_transformation_matrix().
        """
        fft_peaks = np.array(self.clicked_peaks)
        if fft_peaks.shape == (2, 2):
            if self.align_x_axis:
                new_fft1 = np.array(
                    [0, 4.065 * 2 / np.sqrt(3)]
                )  # Forces one of the vectors to be on the x axis

            else:
                fft_length1 = np.linalg.norm(
                    fft_peaks[0]
                )  # Distance of the first fft peak to the origin
                # This line first normalizes the vector then scales it by the expected length in 1/nm to determine the first transformed fft vector.
                new_fft1 = (
                    fft_peaks[0] / fft_length1 * 4.065 * 2 / np.sqrt(3)
                )  # 4.065 is 1/.246 where .246 is the graphene lattice constant in nm.

            new_fft2 = (
                np.array([[0.5, np.sqrt(3) / 2], [-np.sqrt(3) / 2, 0.5]]) @ new_fft1
            )  # Multiplies new_fft1 by a 60 degree rotation matrix to get the second reciprocal lattice vector.
            return np.c_[new_fft1, new_fft2] @ np.linalg.inv(fft_peaks.T)

        else:
            print("Something is fucked up!!! Your fft_peak_positions matrix isn't 2x2.")
            return np.array(
                [[1, 0], [0, 1]]
            )  # Return identity matrix if fft_peaks has the wrong shape

    def get_real_space_transformation_matrix(self):
        """
        Returns the inverse transpose of get_fft_transformation_matrix().
        This can be applied to the x and y coordinates of real space images to
        properly calibrate them.
        """
        return np.linalg.inv(self.get_fft_transformation_matrix().T)

    @property
    def real_space_transformation_matrix(self):
        return self.get_real_space_transformation_matrix()

    @property
    def fft_transformation_matrix(self):
        return self.get_fft_transformation_matrix()

    def get_measured_lattice_constants(self):
        return [
            1 / np.linalg.norm(vector) * 2 / np.sqrt(3) for vector in self.clicked_peaks
        ]

    def plot_transformed_image(self, transformation_matrix):
        """
        Plots the real space image transformed using transformation_matrix in the
        bottom left panel.

        Parameters
        ----------
        transformation_matrix : NDArray
            The matrix used to transform the x and y coordinates of the data.
            Must have shape (2, 2).
        """
        x, y = np.mgrid[
            0 : self.sxm_data.x_range : (self.sxm_data.x_pixels + 1) * 1j,
            0 : self.sxm_data.y_range : (self.sxm_data.y_pixels + 1) * 1j,
        ]
        transformed_x, transformed_y = transform_coordinates(
            x, y, transformation_matrix
        )

        self.transformed_real_ax.set_xlim(
            [np.amin(transformed_x), np.amax(transformed_x)]
        )
        self.transformed_real_ax.set_ylim(
            [np.amin(transformed_y), np.amax(transformed_y)]
        )
        self.transformed_im_plot = self.transformed_real_ax.pcolormesh(
            transformed_x,
            transformed_y,
            self.image_data,
            cmap=self.cmap,
            rasterized=True,
        )
        self.fig.canvas.draw()

    def plot_transformed_fft(self, transformation_matrix):
        """
        Plots the FFT transformed using transformation_matrix in the bottom right
        panel.

        Parameters
        ----------
        transformation_matrix : NDArray
            The matrix used to transform the kx and ky coordinates of the FFT.
            Must have shape (2, 2).
        """
        fft_x_range = self.sxm_data.x_pixels / self.sxm_data.x_range
        fft_y_range = self.sxm_data.y_pixels / self.sxm_data.y_range
        fft_x, fft_y = np.mgrid[
            -fft_x_range / 2 : fft_x_range / 2 : (self.sxm_data.x_pixels + 1) * 1j,
            -fft_y_range / 2 : fft_y_range / 2 : (self.sxm_data.y_pixels + 1) * 1j,
        ]

        transformed_x, transformed_y = transform_coordinates(
            fft_x, fft_y, transformation_matrix
        )

        self.transformed_fft_ax.set_xlim(
            [np.amin(transformed_x), np.amax(transformed_x)]
        )
        self.transformed_fft_ax.set_ylim(
            [np.amin(transformed_y), np.amax(transformed_y)]
        )
        self.transformed_fft_plot = self.transformed_fft_ax.pcolormesh(
            transformed_x,
            transformed_y,
            self.fft_image_data,
            cmap=self.cmap,
            clim=self.fft_clim,
            rasterized=True,
        )
        self.fig.canvas.draw()

    def clear_transformed_plots(self):
        self.transformed_real_ax.cla()
        self.transformed_fft_ax.cla()

        self.transformed_im_plot = None
        self.transformed_fft_plot = None

    def _on_button_press(self, event: matplotlib.backend_bases.MouseEvent):
        """
        Handles mouse press events.
        """
        # Right click in fft window
        if event.inaxes == self.fft_ax:
            if event.button == 1:
                if len(self.clicked_peaks) < 2:
                    cont, ind = self.fft_peak_scatter.contains(event)
                    if cont:
                        index = ind["ind"][0]
                        self.clicked_peaks.append(self.fft_peaks[index])
                        new_facecolors = self.fft_peak_scatter.get_facecolor()
                        new_facecolors[index] = (1, 0, 0, 1)
                        self.fft_peak_scatter.set_facecolor(new_facecolors)
                        self.fft_peak_scatter.set_edgecolor(new_facecolors)
                        self.fig.canvas.draw()

                        # Draw the transformed images if two peaks have been clicked
                        if len(self.clicked_peaks) == 2:
                            self.plot_transformed_image(
                                self.get_real_space_transformation_matrix()
                            )
                            self.plot_transformed_fft(
                                self.get_fft_transformation_matrix()
                            )

            elif event.button == 3:
                self.clicked_peaks = []
                self.clear_transformed_plots()
                self.fft_peak_scatter = self.fft_ax.scatter(
                    self.fft_peak_locations[:, 0],
                    self.fft_peak_locations[:, 1],
                    color=[[0.12156863, 0.46666667, 0.70588235, 1.0]]
                    * len(self.fft_peak_locations),
                )
                self.fig.canvas.draw()
