# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.utils.compat.optional_deps import HAS_PLT
from astropy.visualization.interval import ManualInterval, PercentileInterval
from astropy.visualization.mpl_normalize import (
    AsinhNorm,
    ImageNormalize,
    SimpleNorm,
    imshow_norm,
    imshow_simple_norm,
    simple_norm,
)
from astropy.visualization.stretch import LogStretch, PowerStretch, SqrtStretch

DATA = np.linspace(0.0, 15.0, 6)
DATA2 = np.arange(3)
DATA2SCL = 0.5 * DATA2
DATA3 = np.linspace(-3.0, 3.0, 7)
STRETCHES = (SqrtStretch(), PowerStretch(0.5), LogStretch())
INVALID = (None, -np.inf, -1)


@pytest.mark.skipif(HAS_PLT, reason="matplotlib is installed")
def test_normalize_error_message():
    with pytest.raises(
        ImportError, match=r"matplotlib is required in order to use this class."
    ):
        ImageNormalize()


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
class TestNormalize:
    def test_invalid_interval(self):
        with pytest.raises(TypeError):
            ImageNormalize(vmin=2.0, vmax=10.0, interval=ManualInterval, clip=True)

    def test_invalid_vmin_vmax(self):
        with pytest.raises(ValueError):
            norm = ImageNormalize(vmin=10.0, vmax=2.0)
            norm(10)

    def test_invalid_stretch(self):
        with pytest.raises(TypeError):
            ImageNormalize(vmin=2.0, vmax=10.0, stretch=SqrtStretch, clip=True)

    def test_stretch_none(self):
        with pytest.raises(ValueError):
            ImageNormalize(vmin=2.0, vmax=10.0, stretch=None)

    def test_scalar(self):
        norm = ImageNormalize(vmin=2.0, vmax=10.0, stretch=SqrtStretch(), clip=True)
        norm2 = ImageNormalize(
            data=6, interval=ManualInterval(2, 10), stretch=SqrtStretch(), clip=True
        )
        assert_allclose(norm(6), 0.70710678)
        assert_allclose(norm(6), norm2(6))

    def test_vmin_vmax_equal(self):
        norm = ImageNormalize(vmin=2.0, vmax=2.0)
        data = np.arange(10) - 5.0
        assert_array_equal(norm(data), 0)

    def test_clip(self):
        norm = ImageNormalize(vmin=2.0, vmax=10.0, stretch=SqrtStretch(), clip=True)
        norm2 = ImageNormalize(
            DATA, interval=ManualInterval(2, 10), stretch=SqrtStretch(), clip=True
        )
        output = norm(DATA)
        expected = [0.0, 0.35355339, 0.70710678, 0.93541435, 1.0, 1.0]
        assert_allclose(output, expected)
        assert_allclose(output.mask, [0, 0, 0, 0, 0, 0])
        assert_allclose(output, norm2(DATA))

    def test_noclip(self):
        norm = ImageNormalize(
            vmin=2.0, vmax=10.0, stretch=SqrtStretch(), clip=False, invalid=None
        )
        norm2 = ImageNormalize(
            DATA,
            interval=ManualInterval(2, 10),
            stretch=SqrtStretch(),
            clip=False,
            invalid=None,
        )
        output = norm(DATA)
        expected = [np.nan, 0.35355339, 0.70710678, 0.93541435, 1.11803399, 1.27475488]
        assert_allclose(output, expected)
        assert_allclose(output.mask, [0, 0, 0, 0, 0, 0])
        assert_allclose(norm.inverse(norm(DATA))[1:], DATA[1:])
        assert_allclose(output, norm2(DATA))

    def test_implicit_autoscale(self):
        norm = ImageNormalize(vmin=None, vmax=10.0, stretch=SqrtStretch(), clip=False)
        norm2 = ImageNormalize(
            DATA, interval=ManualInterval(None, 10), stretch=SqrtStretch(), clip=False
        )
        output = norm(DATA)
        assert norm.vmin == np.min(DATA)
        assert norm.vmax == 10.0
        assert_allclose(output, norm2(DATA))

        norm = ImageNormalize(vmin=2.0, vmax=None, stretch=SqrtStretch(), clip=False)
        norm2 = ImageNormalize(
            DATA, interval=ManualInterval(2, None), stretch=SqrtStretch(), clip=False
        )
        output = norm(DATA)
        assert norm.vmin == 2.0
        assert norm.vmax == np.max(DATA)
        assert_allclose(output, norm2(DATA))

    def test_call_clip(self):
        """Test that the clip keyword is used when calling the object."""
        data = np.arange(5)
        norm = ImageNormalize(vmin=1.0, vmax=3.0, clip=False)

        output = norm(data, clip=True)
        assert_equal(output.data, [0, 0, 0.5, 1.0, 1.0])
        assert np.all(~output.mask)

        output = norm(data, clip=False)
        assert_equal(output.data, [-0.5, 0, 0.5, 1.0, 1.5])
        assert np.all(~output.mask)

    def test_masked_clip(self):
        mdata = ma.array(DATA, mask=[0, 0, 1, 0, 0, 0])
        norm = ImageNormalize(vmin=2.0, vmax=10.0, stretch=SqrtStretch(), clip=True)
        norm2 = ImageNormalize(
            mdata, interval=ManualInterval(2, 10), stretch=SqrtStretch(), clip=True
        )
        output = norm(mdata)
        expected = [0.0, 0.35355339, 1.0, 0.93541435, 1.0, 1.0]
        assert_allclose(output.filled(-10), expected)
        assert_allclose(output.mask, [0, 0, 0, 0, 0, 0])
        assert_allclose(output, norm2(mdata))

    def test_masked_noclip(self):
        mdata = ma.array(DATA, mask=[0, 0, 1, 0, 0, 0])
        norm = ImageNormalize(
            vmin=2.0, vmax=10.0, stretch=SqrtStretch(), clip=False, invalid=None
        )
        norm2 = ImageNormalize(
            mdata,
            interval=ManualInterval(2, 10),
            stretch=SqrtStretch(),
            clip=False,
            invalid=None,
        )
        output = norm(mdata)
        expected = [np.nan, 0.35355339, -10, 0.93541435, 1.11803399, 1.27475488]
        assert_allclose(output.filled(-10), expected)
        assert_allclose(output.mask, [0, 0, 1, 0, 0, 0])

        assert_allclose(norm.inverse(norm(DATA))[1:], DATA[1:])
        assert_allclose(output, norm2(mdata))

    def test_invalid_data(self):
        data = np.arange(25.0).reshape((5, 5))
        data[2, 2] = np.nan
        data[1, 2] = np.inf
        percent = 85.0
        interval = PercentileInterval(percent)

        # initialized without data
        norm = ImageNormalize(interval=interval)
        norm(data)  # sets vmin/vmax
        assert_equal((norm.vmin, norm.vmax), (1.65, 22.35))

        # initialized with data
        norm2 = ImageNormalize(data, interval=interval)
        assert_equal((norm2.vmin, norm2.vmax), (norm.vmin, norm.vmax))

        norm3 = simple_norm(data, "linear", percent=percent)
        assert_equal((norm3.vmin, norm3.vmax), (norm.vmin, norm.vmax))

        assert_allclose(norm(data), norm2(data))
        assert_allclose(norm(data), norm3(data))

        norm4 = ImageNormalize()
        norm4(data)  # sets vmin/vmax
        assert_equal((norm4.vmin, norm4.vmax), (0, 24))

        norm5 = ImageNormalize(data)
        assert_equal((norm5.vmin, norm5.vmax), (norm4.vmin, norm4.vmax))

    @pytest.mark.parametrize("stretch", STRETCHES)
    def test_invalid_keyword(self, stretch):
        norm1 = ImageNormalize(
            stretch=stretch, vmin=-1, vmax=1, clip=False, invalid=None
        )
        norm2 = ImageNormalize(stretch=stretch, vmin=-1, vmax=1, clip=False)
        norm3 = ImageNormalize(
            DATA3, stretch=stretch, vmin=-1, vmax=1, clip=False, invalid=-1.0
        )
        result1 = norm1(DATA3)
        result2 = norm2(DATA3)
        result3 = norm3(DATA3)
        assert_equal(result1[0:2], (np.nan, np.nan))
        assert_equal(result2[0:2], (-1.0, -1.0))
        assert_equal(result1[2:], result2[2:])
        assert_equal(result2, result3)


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
class TestImageScaling:
    def test_linear(self):
        """Test linear scaling."""
        norm = simple_norm(DATA2, stretch="linear")
        assert_allclose(norm(DATA2), DATA2SCL, atol=0, rtol=1.0e-5)

    def test_sqrt(self):
        """Test sqrt scaling."""
        norm1 = simple_norm(DATA2, stretch="sqrt")
        assert_allclose(norm1(DATA2), np.sqrt(DATA2SCL), atol=0, rtol=1.0e-5)

    @pytest.mark.parametrize("invalid", INVALID)
    def test_sqrt_invalid_kw(self, invalid):
        stretch = SqrtStretch()
        norm1 = simple_norm(
            DATA3, stretch="sqrt", vmin=-1, vmax=1, clip=False, invalid=invalid
        )
        norm2 = ImageNormalize(
            stretch=stretch, vmin=-1, vmax=1, clip=False, invalid=invalid
        )
        assert_equal(norm1(DATA3), norm2(DATA3))

    def test_power(self):
        """Test power scaling."""
        power = 3.0
        norm = simple_norm(DATA2, stretch="power", power=power)
        assert_allclose(norm(DATA2), DATA2SCL**power, atol=0, rtol=1.0e-5)

    def test_log(self):
        """Test log10 scaling."""
        norm = simple_norm(DATA2, stretch="log")
        ref = np.log10(1000 * DATA2SCL + 1.0) / np.log10(1001.0)
        assert_allclose(norm(DATA2), ref, atol=0, rtol=1.0e-5)

    def test_log_with_log_a(self):
        """Test log10 scaling with a custom log_a."""
        log_a = 100
        norm = simple_norm(DATA2, stretch="log", log_a=log_a)
        ref = np.log10(log_a * DATA2SCL + 1.0) / np.log10(log_a + 1)
        assert_allclose(norm(DATA2), ref, atol=0, rtol=1.0e-5)

    def test_asinh(self):
        """Test asinh scaling."""
        norm = simple_norm(DATA2, stretch="asinh")
        ref = np.arcsinh(10 * DATA2SCL) / np.arcsinh(10)
        assert_allclose(norm(DATA2), ref, atol=0, rtol=1.0e-5)

    def test_asinh_with_asinh_a(self):
        """Test asinh scaling with a custom asinh_a."""
        asinh_a = 0.5
        norm = simple_norm(DATA2, stretch="asinh", asinh_a=asinh_a)
        ref = np.arcsinh(DATA2SCL / asinh_a) / np.arcsinh(1.0 / asinh_a)
        assert_allclose(norm(DATA2), ref, atol=0, rtol=1.0e-5)

    def test_asinh_with_asinh_x0(self):
        """Test asinh scaling with a custom asinh_x0."""
        asinh_a = 0.5
        asinh_x0 = 0.2
        norm = simple_norm(DATA2, stretch="asinh", asinh_a=asinh_a, asinh_x0=asinh_x0)
        assert norm.stretch.x0 == asinh_x0
        offset = np.arcsinh(asinh_x0 / asinh_a)
        denom = np.arcsinh((1.0 - asinh_x0) / asinh_a) + offset
        ref = (np.arcsinh((DATA2SCL - asinh_x0) / asinh_a) + offset) / denom
        assert_allclose(norm(DATA2), ref, atol=0, rtol=1.0e-5)

    def test_asinh_x0_sky_zero_point(self):
        """
        Test an asinh stretch with its zero point at the sky and the
        black point below the sky.
        """
        vmin = -1.0421906109874948  # arcsinh(vmin / 2) = -0.5
        vmax = 377.62182631705866
        asinh_a = 2.0 / (vmax - vmin)
        asinh_x0 = -vmin / (vmax - vmin)
        data = np.linspace(vmin, vmax, 1001)
        norm = simple_norm(
            data,
            stretch="asinh",
            vmin=vmin,
            vmax=vmax,
            asinh_a=asinh_a,
            asinh_x0=asinh_x0,
        )
        ref = (np.arcsinh(data / 2) + 0.5) / (np.arcsinh(vmax / 2) + 0.5)
        assert_allclose(norm(data), ref, atol=1e-12, rtol=0)

    def test_sinh(self):
        """Test sinh scaling."""
        sinh_a = 0.5
        norm = simple_norm(DATA2, stretch="sinh", sinh_a=sinh_a)
        ref = np.sinh(DATA2SCL / sinh_a) / np.sinh(1 / sinh_a)
        assert_allclose(norm(DATA2), ref, atol=0, rtol=1.0e-5)

    def test_sinh_with_sinh_x0(self):
        """Test sinh scaling with a custom sinh_x0."""
        sinh_a = 0.5
        sinh_x0 = 0.2
        norm = simple_norm(DATA2, stretch="sinh", sinh_a=sinh_a, sinh_x0=sinh_x0)
        assert norm.stretch.x0 == sinh_x0
        offset = np.sinh(sinh_x0 / sinh_a)
        denom = np.sinh((1.0 - sinh_x0) / sinh_a) + offset
        ref = (np.sinh((DATA2SCL - sinh_x0) / sinh_a) + offset) / denom
        assert_allclose(norm(DATA2), ref, atol=0, rtol=1.0e-5)

    def test_min(self):
        """Test linear scaling."""
        norm = simple_norm(DATA2, stretch="linear", vmin=1.0, clip=True)
        assert_allclose(norm(DATA2), [0.0, 0.0, 1.0], atol=0, rtol=1.0e-5)

    def test_percent(self):
        """Test percent keywords."""
        norm = simple_norm(DATA2, stretch="linear", percent=99.0, clip=True)
        assert_allclose(norm(DATA2), DATA2SCL, atol=0, rtol=1.0e-5)

        norm2 = simple_norm(
            DATA2, stretch="linear", min_percent=0.5, max_percent=99.5, clip=True
        )
        assert_allclose(norm(DATA2), norm2(DATA2), atol=0, rtol=1.0e-5)

    def test_invalid_stretch(self):
        """Test invalid stretch keyword."""
        with pytest.raises(ValueError):
            simple_norm(DATA2, stretch="invalid")


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
@pytest.mark.parametrize("stretch", ["linear", "sqrt", "power", "log", "asinh", "sinh"])
def test_simplenorm(stretch):
    data = np.arange(25).reshape((5, 5))
    snorm = SimpleNorm(stretch, percent=99)
    norm = snorm(data)
    assert isinstance(norm, ImageNormalize)
    assert_allclose(norm(data), simple_norm(data, stretch, percent=99)(data))


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
@pytest.mark.parametrize("stretch", ["asinh", "sinh"])
def test_simplenorm_x0(stretch):
    data = np.arange(25).reshape((5, 5))
    kwargs = {f"{stretch}_a": 0.3, f"{stretch}_x0": 0.1}
    snorm = SimpleNorm(stretch, **kwargs)
    assert snorm.stretch.a == kwargs[f"{stretch}_a"]
    assert snorm.stretch.x0 == kwargs[f"{stretch}_x0"]
    assert_allclose(snorm(data)(data), simple_norm(data, stretch, **kwargs)(data))


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
def test_simplenorm_imshow():
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

    data = np.arange(25).reshape((5, 5))
    fig = Figure()
    ax = fig.add_subplot()
    snorm = SimpleNorm("sqrt", percent=99)
    axim = snorm.imshow(data, ax=ax)
    assert isinstance(axim, AxesImage)
    keys = ("vmin", "vmax", "stretch", "clip", "invalid")
    for key in keys:
        assert getattr(axim.norm, key) == getattr(snorm(data), key)

    fig.clear()
    axim = snorm.imshow(data, ax=None)

    with pytest.raises(ValueError):
        snorm.imshow(data, ax=ax, norm=ImageNormalize())


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
def test_imshow_norm():
    from matplotlib.figure import Figure

    image = np.random.randn(10, 10)

    fig = Figure()
    ax = fig.add_subplot(label="test_imshow_norm")
    imshow_norm(image, ax=ax)

    with pytest.raises(ValueError):
        # illegal to manually pass in normalization since that defeats the point
        imshow_norm(image, ax=ax, norm=ImageNormalize())

    fig.clear()
    imshow_norm(image, ax=ax, vmin=0, vmax=1)

    # make sure the matplotlib version works
    fig.clear()
    imres, norm = imshow_norm(image, ax=None)

    assert isinstance(norm, ImageNormalize)


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
def test_imshow_simple_norm():
    from matplotlib.figure import Figure

    image = np.arange(100).reshape(10, 10)

    fig = Figure()
    ax = fig.add_subplot(label="test_imshow_simple_norm")
    imshow_simple_norm(image, ax=ax)

    with pytest.raises(ValueError):
        # illegal to manually pass in normalization since that defeats the point
        imshow_simple_norm(image, ax=ax, norm=ImageNormalize())

    fig.clear()
    imshow_simple_norm(image, ax=ax, vmin=0, vmax=1)

    fig.clear()
    imshow_simple_norm(image, ax=ax, min_percent=1, max_percent=99.9, stretch="sinh")

    # make sure the matplotlib version works
    fig.clear()
    imres, norm = imshow_simple_norm(image, ax=None)

    assert np.all(imres.get_array() == image)

    # ensure the normalization is *not* just minmax like default imshow
    assert (image.min(), image.max()) == imres.get_clim()

    assert isinstance(norm, ImageNormalize)


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
def test_norm_without_data():
    from matplotlib.figure import Figure

    image = np.arange(10).reshape((1, 10))
    interval = ManualInterval(2, 5)
    norm_without_data = ImageNormalize(interval=interval)

    assert norm_without_data.vmin is None
    assert norm_without_data.vmax is None

    fig = Figure()
    ax = fig.add_subplot()
    ax.imshow(image, norm=norm_without_data)  # calls norm_without_data.autoscale_None()

    assert norm_without_data.vmin == interval.vmin
    assert norm_without_data.vmax == interval.vmax


@pytest.fixture(name="noise_image")
def fixture_noise_image():
    """
    Return an image with a background of 10, noise of 2, and a few
    bright pixels.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 2.0, (100, 100))
    data[0, :5] = 1000.0
    return data


@pytest.mark.skipif(not HAS_PLT, reason="requires matplotlib")
class TestAsinhNorm:
    def test_known_values(self):
        data = np.linspace(-5.0, 120.0, 51)
        norm = AsinhNorm(softening=2.0, min_nsigma=-1.0)(
            data, background=5.0, noise=1.0, vmax=100.0
        )
        assert isinstance(norm, ImageNormalize)
        assert norm.vmin == 4.0
        assert norm.vmax == 100.0
        assert_allclose(norm.stretch.a, 2.0 / 96.0)
        assert_allclose(norm.stretch.x0, 1.0 / 96.0)

        offset = np.arcsinh(0.5)
        ref = (np.arcsinh((data - 5.0) / 2.0) + offset) / (
            np.arcsinh(95.0 / 2.0) + offset
        )
        result = norm(data)
        mask = (data >= 4.0) & (data <= 100.0)
        assert_allclose(result[mask], ref[mask], atol=1e-12)
        assert np.all(result[data < 4.0] < 0)
        assert np.all(result[data > 100.0] > 1)

    def test_sky_subtracted(self):
        """
        Test that a sky-subtracted image with the zero point at the
        background reproduces arcsinh(data / 2), with the black point
        at -0.5 in stretched units.
        """
        vmin = -1.0421906109874948  # arcsinh(vmin / 2) = -0.5
        vmax = 377.62182631705866
        data = np.linspace(vmin, vmax, 1001)
        norm = AsinhNorm(softening=2.0, min_nsigma=vmin)(
            data, background=0.0, noise=1.0, vmax=vmax
        )
        ref = (np.arcsinh(data / 2) + 0.5) / (np.arcsinh(vmax / 2) + 0.5)
        assert_allclose(norm(data), ref, atol=1e-12, rtol=0)

    def test_estimated_values(self, noise_image):
        _, median, std = sigma_clipped_stats(noise_image, sigma=3.0)
        vmax = np.percentile(noise_image, 99.7)
        norm = AsinhNorm()(noise_image)
        assert_allclose(norm.vmin, median - std, rtol=1e-12)
        assert_allclose(norm.vmax, vmax, rtol=1e-12)
        span = vmax - (median - std)
        assert_allclose(norm.stretch.a, 2.0 * std / span, rtol=1e-12)
        assert_allclose(norm.stretch.x0, std / span, rtol=1e-12)
        assert_allclose(median, 10.0, atol=0.05)
        assert_allclose(std, 2.0, atol=0.05)

    def test_partial_override(self, noise_image):
        _, median, std = sigma_clipped_stats(noise_image, sigma=3.0)
        norm = AsinhNorm()(noise_image, background=0.0)
        assert_allclose(norm.vmin, -std, rtol=1e-12)
        norm = AsinhNorm()(noise_image, noise=5.0)
        assert_allclose(norm.vmin, median - 5.0, rtol=1e-12)

    def test_sigma_clip(self, noise_image):
        _, median, std = sigma_clipped_stats(noise_image, sigma=2.0, maxiters=10)
        norm = AsinhNorm(sigma_clip=SigmaClip(sigma=2.0, maxiters=10))
        assert_allclose(norm(noise_image).vmin, median - std, rtol=1e-12)

    def test_reuse(self, noise_image):
        norm = AsinhNorm(min_nsigma=-2.0)
        norm1 = norm(noise_image)
        norm2 = norm(3.0 * noise_image)
        assert_allclose(norm2.vmin, 3.0 * norm1.vmin, rtol=1e-12)
        assert_allclose(norm2.vmax, 3.0 * norm1.vmax, rtol=1e-12)
        assert_allclose(norm2.stretch.a, norm1.stretch.a, rtol=1e-12)
        assert_allclose(norm2.stretch.x0, norm1.stretch.x0, rtol=1e-12)

    def test_invalid_data_values(self, noise_image):
        data = noise_image.copy()
        data[1, :10] = np.nan
        data[2, :10] = np.inf
        masked = ma.MaskedArray(noise_image, mask=np.zeros(data.shape, bool))
        masked.mask[1:3, :10] = True
        norm1 = AsinhNorm()(data)
        norm2 = AsinhNorm()(masked)
        _, median, std = sigma_clipped_stats(masked, sigma=3.0)
        assert_allclose(norm1.vmin, median - std, rtol=1e-12)
        assert_allclose(norm2.vmin, median - std, rtol=1e-12)
        assert_allclose(norm1.vmax, norm2.vmax, rtol=1e-12)

    def test_max_percent(self, noise_image):
        norm = AsinhNorm(max_percent=99.0)(noise_image)
        assert_allclose(norm.vmax, np.percentile(noise_image, 99.0), rtol=1e-12)

    def test_clip_invalid(self, noise_image):
        norm = AsinhNorm(clip=True)(noise_image)
        result = norm(noise_image)
        assert result.min() == 0.0
        assert result.max() == 1.0

        norm = AsinhNorm(invalid=None)(noise_image)
        assert norm.invalid is None

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"softening": 0.0}, "softening must be > 0"),
            ({"softening": -1.0}, "softening must be > 0"),
            ({"min_nsigma": 0.5}, "min_nsigma must be <= 0"),
            ({"max_percent": 0.0}, "max_percent must be > 0 and <= 100"),
            ({"max_percent": 101.0}, "max_percent must be > 0 and <= 100"),
        ],
    )
    def test_invalid_init(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            AsinhNorm(**kwargs)

    def test_invalid_call(self, noise_image):
        norm = AsinhNorm()
        with pytest.raises(ValueError, match="finite, unmasked values"):
            norm(np.full((10, 10), np.nan))
        with pytest.raises(ValueError, match="noise must be > 0"):
            norm(np.full((10, 10), 3.0))
        with pytest.raises(ValueError, match="noise must be > 0"):
            norm(noise_image, noise=0.0)
        match = "vmax must be greater than the background"
        with pytest.raises(ValueError, match=match):
            norm(noise_image, vmax=5.0)
        with pytest.raises(ValueError, match=match):
            norm(noise_image, background=2000.0)

    def test_imshow(self, noise_image):
        from matplotlib.figure import Figure
        from matplotlib.image import AxesImage

        fig = Figure()
        ax = fig.add_subplot()
        norm = AsinhNorm()
        axim = norm.imshow(noise_image, ax=ax, background=10.0, origin="lower")
        assert isinstance(axim, AxesImage)
        assert axim.origin == "lower"
        expected = norm(noise_image, background=10.0)
        assert axim.norm.vmin == expected.vmin
        assert axim.norm.vmax == expected.vmax

        with pytest.raises(ValueError, match="already defines the norm"):
            norm.imshow(noise_image, ax=ax, norm=ImageNormalize())
