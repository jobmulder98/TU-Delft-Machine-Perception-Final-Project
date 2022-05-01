import cv2
import IPython.display
import numpy as np
import PIL.Image
from matplotlib import animation
from matplotlib import pyplot as plt


def create_animation(images, interval_ms=100, **fig_kwargs):

    # use larger plot by default
    if "figsize" not in fig_kwargs:
        fig_kwargs["figsize"] = (13, 9)

    fig, ax = plt.subplots(**fig_kwargs)
    fig.tight_layout()
    ax.axis("off")
    im = ax.imshow(images[0][:, :, ::-1])
    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(images[0][:, :, ::-1])  # ::-1: BGR --> RGB

    def animate(i):
        image = images[i]
        if image is not None:
            im.set_data(image[:, :, ::-1])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(images), interval=interval_ms)

    # disable warning for video creation (anim.to_*())
    import logging

    logging.getLogger("matplotlib.animation").disabled = True

    # from IPython.core.display import HTML
    # use HTML(anim.to_html5_video()) to show within jupyter notebook as video
    # or HTML(anim.to_jshtml()) to show within jupyter notebook as interactive widget
    return anim


def draw_bbox_to_image(image, bbox, color=(0, 255, 0), thickness=5):
    corner_coords = bbox.get_bbox_corners_vis()
    image = cv2.rectangle(image, corner_coords[:2], corner_coords[-2:], color, thickness)
    return image


def showimage(a):
    """Show an image below the current jupyter notebook cell.
    Expects gray or bgr input (opencv2 default)"""
    # bgr -> rgb
    if len(a.shape) > 2 and a.shape[2] == 3:
        a = a[..., ::-1]  # bgr -> rgb
    image = PIL.Image.fromarray(a)
    IPython.display.display(image)  # display in cell output


# https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
colors_qualitative = np.array(
    [
        [166, 206, 227],
        [31, 120, 180],
        [178, 223, 138],
        [51, 160, 44],
        [251, 154, 153],
        [227, 26, 28],
        [253, 191, 111],
        [255, 127, 0],
        [202, 178, 214],
        [106, 61, 154],
        [255, 255, 153],
        [177, 89, 40],
    ]
)

# rgb representation for k3d
colors_qualitative_k3d = np.dot(colors_qualitative, np.asarray([2 ** 16, 2 ** 8, 2 ** 0])).tolist()
