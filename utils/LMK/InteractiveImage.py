import numpy as np
import matplotlib as plt


class InteractiveImage:

    def __init__(self, ax, image, lmk_size=3):
        self.ax = ax
        self.img = image
        self.press = None
        self.ols = int(lmk_size/2)  # offset lmk_size
        self.is_occluded = False

        self.ax.imshow(self.img)

    def on_click(self, event):
        self.press = [np.round(event.xdata).astype(int),
                      np.round(event.ydata).astype(int)]

    def on_release(self, event):
        if self.press is not None:
            img = np.copy(self.img)

            lmk_x = self.press[1]
            lmk_y = self.press[0]

            img[(lmk_x-self.ols):(lmk_x+self.ols),
                (lmk_y-self.ols):(lmk_y+self.ols)] = [1, 0, 0]

            self.ax.imshow(img)
            plt.draw()

    def on_enter(self, event):
        if self.press is not None and event.key == 'enter':
            plt.close()
        elif self.press is None and event.key == 'enter':
            print("No Landmark selected!")
        elif self.press is None and event.key == 'o':
            print("Landmark Occluded")
            self.is_occluded = True
            plt.close()
        elif event.key == 'escape':
            plt.close()

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.ax.figure.canvas.mpl_connect(
            'button_press_event', self.on_click)
        self.cidrelease = self.ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidenter = self.ax.figure.canvas.mpl_connect(
            'key_press_event', self.on_enter)

    def disconnect(self):
        """Disconnect all callbacks."""
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidenter)