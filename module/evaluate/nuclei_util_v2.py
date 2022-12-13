import math
import scipy
import numpy as np
import skimage.morphology
import skimage.segmentation
import centrosome.smooth
import centrosome.outline
import centrosome.cpmorphology


class image_class():
    def __init__(self, image):
        self.pixel_data = image
        self.dimensions = len(image.shape)
        self.mask = np.ones_like(image, dtype=bool)
        self.volumetric = False


def identifyprimaryobjects(
        input_image,
        size_range=(10, 40),
        is_threshold=False,
        exclude_border_objects=False,
        exclude_size=True,
):
    res = {}
    image = image_class(input_image)
    basic = True
    '''
    get the global and local threshold value to filter the background
    '''
    local_threshold = global_threshold = skimage.filters.threshold_li(image.pixel_data)
    if (is_threshold):
        local_threshold = global_threshold = skimage.filters.threshold_otsu(image.pixel_data)

    sigma = 1.
    blurred_image = centrosome.smooth.smooth_with_function_and_mask(
        image.pixel_data, lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0), image.mask)
    binary_image = (blurred_image >= local_threshold) & image.mask

    res['sigma'] = sigma

    def size_fn(size, is_foreground):
        return size < size_range[1] * size_range[1]

    '''
    filling the hole in the object
    '''
    binary_image = centrosome.cpmorphology.fill_labeled_holes(binary_image, size_fn=size_fn)
    '''
    labeled the image with integer by connected component
    '''
    labeled_image, object_count = scipy.ndimage.label(binary_image, np.ones((3, 3), bool))

    def smooth_image(image, mask):
        '''
        Apply the smoothing filter to the image
        '''
        filter_size = 2.35 * size_range[0] / 3.5
        s_sigma = filter_size / 2.35
        if filter_size == 0:
            return image
        '''
        We not only want to smooth using a Gaussian, but we want to limit
        the spread of the smoothing to 2 SD, partly to make things happen
        locally, partly to make things run faster, partly to try to match
        the Matlab behavior.
        '''
        filter_size = np.clip(int(float(filter_size) / 2.0), 1, np.inf)
        f = (1 / np.sqrt(2.0 * np.pi) / s_sigma * np.exp(-0.5 * np.arange(-filter_size, filter_size + 1)**2 / s_sigma**2))

        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f, axis=0, mode='constant')
            return scipy.ndimage.convolve1d(output, f, axis=1, mode='constant')

        '''
        Use the trick where you similarly convolve an array of ones to find
        out the edge effects, then divide to correct the edge effects
        '''
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        return masked_image

    def get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
        if image_resize_factor < 1.0:
            shape = np.array(image.shape) * image_resize_factor
            i_j = (np.mgrid[0:shape[0], 0:shape[1]].astype(float) / image_resize_factor)
            resized_image = scipy.ndimage.map_coordinates(image, i_j)
            resized_labels = scipy.ndimage.map_coordinates(labeled_image, i_j, order=0).astype(labeled_image.dtype)
        '''
        find local maxima
        '''
        binary_maxima_image = centrosome.cpmorphology.is_local_maximum(image, labeled_image, maxima_mask)
        binary_maxima_image[image <= 0] = 0

        if image_resize_factor < 1.0:
            inverse_resize_factor = (float(image.shape[0]) / float(binary_maxima_image.shape[0]))
            i_j = (np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(float) / inverse_resize_factor)
            binary_maxima_image = scipy.ndimage.map_coordinates(binary_maxima_image.astype(float), i_j) > .5
            assert (binary_maxima_image.shape[0] == image.shape[0])
            assert (binary_maxima_image.shape[1] == image.shape[1])
        '''
        Erode blobs of touching maxima to a single point
        '''
        shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
        return shrunk_image

    def separate_neighboring_objects(im, labeled_image, object_count):
        '''
        Separate objects based on local maxima or distance transform
        workspace - get the image from here
        labeled_image - image labeled by scipy.ndimage.label
        object_count  - # of objects in image
        returns revised labeled_image, object count, maxima_suppression_size, LoG threshold and filter diameter
        '''
        image = im.pixel_data
        mask = im.mask

        blurred_image = smooth_image(image, mask)

        low_res_maxima = 'Speed up by using lower-resolution image to find local maxima?'
        automatic_suppression = True
        maxima_suppression_size = 7
        if size_range[0] > 10 and (basic or low_res_maxima):
            image_resize_factor = 10.0 / float(size_range[0])
            if not (basic or automatic_suppression):
                maxima_suppression_size = (maxima_suppression_size * image_resize_factor + .5)
        else:
            image_resize_factor = 1.0
            if basic or automatic_suppression:
                maxima_suppression_size = size_range[0] / 1.5

        reported_maxima_suppression_size = maxima_suppression_size / image_resize_factor

        maxima_mask = centrosome.cpmorphology.strel_disk(max(1, maxima_suppression_size - .5))
        '''
        Remove dim maxima
        '''
        maxima_image = get_maxima(blurred_image, labeled_image, maxima_mask, image_resize_factor)
        '''
        Create a marker array where the unlabeled image has a label of
        -(nobjects+1)
        and every local maximum has a unique label which will become
        the object's label. The labels are negative because that
        makes the watershed algorithm use FIFO for the pixels which
        yields fair boundaries when markers compete for pixels.
        '''
        labeled_maxima, object_count = scipy.ndimage.label(maxima_image, np.ones((3, 3), bool))
        '''
        Create the image for watershed
        use the reverse of the image to get valleys at peaks
        '''
        watershed_image = -image

        markers_dtype = (np.int16 if object_count < np.iinfo(np.int16).max else np.int32)
        markers = np.zeros(watershed_image.shape, markers_dtype)
        markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]
        '''
        Some labels have only one maker in them, some have multiple and
        will be split up.
        '''
        watershed_boundaries = skimage.segmentation.watershed(connectivity=np.ones((3, 3), bool),
                                                            image=watershed_image,
                                                            markers=markers,
                                                            mask=labeled_image != 0)

        watershed_boundaries = -watershed_boundaries

        return watershed_boundaries, object_count, reported_maxima_suppression_size

    '''
    separate neighboring objects by watershed and markers of object maxima as center of pool
    '''
    labeled_image, object_count, maxima_suppression_size = separate_neighboring_objects(image, labeled_image, object_count)
    '''
    Filter out objects touching the border or mask
    '''
    def filter_on_border(image, labeled_image):
        '''
        Filter out objects touching the border

        In addition, if the image has a mask, filter out objects
        touching the border of the mask.
        '''
        if exclude_border_objects:
            border_labels = list(labeled_image[0, :])
            border_labels.extend(labeled_image[:, 0])
            border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
            border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
            border_labels = np.array(border_labels)
            '''
            the following histogram has a value > 0 for any object
            with a border pixel
            '''
            histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape), (border_labels, np.zeros(border_labels.shape))),
                                                shape=(np.max(labeled_image) + 1, 1)).todense()
            histogram = np.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
            else:
                '''
                The assumption here is that, if nothing touches the border,
                the mask is a large, elliptical mask that tells you where the
                well is. That's the way the old Matlab code works and it's duplicated here.
                The operation below gets the mask pixels that are on the border of the mask
                The erosion turns all pixels touching an edge to zero. The not of this
                is the border + formerly masked-out pixels.
                '''
                mask_border = np.logical_not(scipy.ndimage.binary_erosion(image.mask))
                mask_border = np.logical_and(mask_border, image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = scipy.sparse.coo_matrix(
                    (np.ones(border_labels.shape), (border_labels, np.zeros(border_labels.shape))),
                    shape=(np.max(labeled_image) + 1, 1)).todense()
                histogram = np.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    labeled_image[histogram_image > 0] = 0
        return labeled_image

    if exclude_border_objects:
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = filter_on_border(image, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0
    '''
    Filter out small and large objects
    '''
    def filter_on_size(labeled_image, object_count):
        '''
        Filter the labeled image based on the size range

        labeled_image - pixel image labels
        object_count - # of objects in the labeled image
        returns the labeled image, and the labeled image with the
        small objects removed
        '''
        if exclude_size and object_count > 0:
            areas = scipy.ndimage.measurements.sum(np.ones(labeled_image.shape), labeled_image,
                                                   np.array(range(0, object_count + 1), dtype=np.int32))
            areas = np.array(areas, dtype=int)
            min_allowed_area = np.pi * (size_range[0] * size_range[0]) / 4
            max_allowed_area = np.pi * (size_range[1] * size_range[1]) / 4
            '''
            area_image has the area of the object at every pixel within the object
            '''
            area_image = areas[labeled_image]
            labeled_image[area_image < min_allowed_area] = 0
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
        else:
            small_removed_labels = labeled_image.copy()
        return labeled_image, small_removed_labels

    if exclude_size:
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = filter_on_size(labeled_image, object_count)
        res['small_removed_labels'] = small_removed_labels
        size_excluded_labeled_image[labeled_image > 0] = 0
    '''
    Fill holes again after watershed
    '''
    labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)
    '''
    Relabel the image
    '''
    labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)
    outline_image = centrosome.outline.outline(labeled_image)

    return object_count, outline_image, labeled_image
