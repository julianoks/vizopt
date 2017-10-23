## adapted from https://matplotlib.org/examples/api/radar_chart.html

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    theta += np.pi/2
    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')
    def draw_circle_patch(self):
        return plt.Circle((0.5, 0.5), 0.5)
    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)
    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1
        draw_patch = patch_dict[frame]
        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        def _gen_axes_patch(self):
            return self.draw_patch()
        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            verts.append(verts[0])
            path = Path(verts)
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}
    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def labels_to_colors(labels):
    cmap = plt.get_cmap('viridis')
    color_dict = {}
    unique_labels = np.unique(labels)
    for i, v in enumerate(unique_labels):
        color_dict[v] = cmap(i / len(unique_labels))
    colors = [color_dict[l] for l in labels]
    return colors

def radar_chart(data, labels, show_axis=False, fill_polygon=False):
    theta = radar_factory(len(data[0]), frame='circle')
    colors = labels_to_colors(labels)
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(projection='radar'), facecolor='white')
    ax.axis('on' if show_axis else 'off')
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    for record, color in zip(data, colors):
        ax.plot(theta, record, color=color)
        if fill_polygon:
            ax.fill(theta, record, facecolor=color, alpha=0.25)
    return fig, ax
