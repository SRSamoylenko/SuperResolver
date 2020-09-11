import numpy as np
from util import Coordinate, cartesian_to_polar, zero_division


class PointSource:
    Statistics = {
        'exponential': np.random.exponential,
        'poisson': np.random.poisson
    }

    def __init__(self, coordinates=[], mean_photons=None, statistics='exponential', field_shape=None):
        self.coordinates = np.array(coordinates) if np.array(coordinates).size != 0 else np.random.random_sample(2)
        self.mean_photons = mean_photons if mean_photons else np.random.random_sample() * 5
        self.radiation_shape = None
        self.field_shape = field_shape
        if field_shape:
            self.define_radiation_shape(field_shape)
        self.statistics = statistics
        self.size = None

    def radiate(self):
        if self.radiation_shape is None:
            raise Exception("Radiation shape hasn't been defined.")
        return self.get_photon_number() * self.radiation_shape

    def get_photon_number(self):
        return self.Statistics[self.statistics](self.mean_photons, size=self.size)

    def define_radiation_shape(self, field_shape):
        res = np.zeros(field_shape).T
        res[int(self.coordinates[Coordinate.Y] * field_shape[Coordinate.Y]),
            int(self.coordinates[Coordinate.X] * field_shape[Coordinate.X])] = 1
        self.radiation_shape = res


class PlanarSource(PointSource):
    RadiationType = {
        'uniform': True,
        'non-uniform': False
    }

    def __init__(self, radiation_type='uniform', **kwargs):
        self.radiation_type = radiation_type
        self.size = None if self.RadiationType[self.radiation_type] else self.field_shape
        PointSource.__init__(self, **kwargs)


class CircularSource(PlanarSource):
    def __init__(self, radius=None, **kwargs):
        self.radius = radius if radius else np.random.random_sample() / 4
        PlanarSource.__init__(self, **kwargs)

    def define_radiation_shape(self, field_shape):
        x, y = make_grid(field_shape)
        r, phi = cartesian_to_polar((x - self.coordinates[Coordinate.X]) * x.shape[Coordinate.Y] / np.max(x.shape),
                                    (y - self.coordinates[Coordinate.Y]) * x.shape[Coordinate.X] / np.max(x.shape))
        self.radiation_shape = np.heaviside((self.radius - r), 0)


class RectangularSource(PlanarSource):
    def __init__(self, lengths=None, **kwargs):
        self.lengths = lengths if lengths else np.random.random_sample(2)
        PlanarSource.__init__(self, **kwargs)

    def define_radiation_shape(self, field_shape):
        x, y = make_grid(field_shape)
        upper_left_corner_coordinates = (self.coordinates[Coordinate.X] - self.lengths[Coordinate.X] / 2,
                                         self.coordinates[Coordinate.Y] - self.lengths[Coordinate.Y] / 2)
        lower_right_corner_coordinates = (self.coordinates[Coordinate.X] + self.lengths[Coordinate.X] / 2,
                                          self.coordinates[Coordinate.Y] + self.lengths[Coordinate.Y] / 2)
        self.radiation_shape = np.heaviside(((upper_left_corner_coordinates[Coordinate.X] - x) *
                                             (x - lower_right_corner_coordinates[Coordinate.X])), 0) * \
                               np.heaviside(((upper_left_corner_coordinates[Coordinate.Y] - y) *
                                             (y - lower_right_corner_coordinates[Coordinate.Y])), 0)


class TriangularSource(PlanarSource):
    def __init__(self, coordinates=[], **kwargs):
        coordinates = np.array(coordinates) if np.array(coordinates).size != 0 else np.random.random_sample((3, 2))
        PlanarSource.__init__(self, coordinates=coordinates, **kwargs)

    def define_radiation_shape(self, field_shape):
        res = np.ones(field_shape).T
        x, y = make_grid(field_shape)
        for i, point in enumerate(self.coordinates):
            line_points = np.delete(self.coordinates, i, 0)
            line_from_points = calculate_line(line_points)
            res *= np.heaviside((line_from_points((x, y)) * line_from_points(point)), 0)
        self.radiation_shape = res


class RotationalFigureSource(PlanarSource):
    def __init__(self, radiuses=None, **kwargs):
        if not radiuses:
            radiuses = np.array([])
            for i in np.arange(int(np.floor(np.random.random_sample() * 10))):
                if i == 0:
                    radiuses = np.append(radiuses, np.random.random_sample() / 50)
                else:
                    radiuses = np.append(radiuses, radiuses[i - 1] + np.random.random_sample() / 50)
        self.radiuses = np.sort(radiuses)[::-1]
        PlanarSource.__init__(self, **kwargs)

    def define_radiation_shape(self, field_shape):
        x, y = make_grid(field_shape)
        r, phi = cartesian_to_polar((x - self.coordinates[Coordinate.X]) * x.shape[Coordinate.Y] / np.max(x.shape),
                                    (y - self.coordinates[Coordinate.Y]) * x.shape[Coordinate.X] / np.max(x.shape))
        res = np.zeros(field_shape).T
        for i, radius in enumerate(self.radiuses):
            if i % 2 == 0:
                res += np.heaviside((radius - r), 0)
            else:
                res -= np.heaviside((radius - r), 0)
        self.radiation_shape = res


class PolygonalSource(PlanarSource):
    def __init__(self, scale=None, rotation=None, n_vertex=None, **kwargs):
        self.n_vertex = n_vertex if n_vertex else 3 + np.floor(np.random.random_sample() * 8) / 2
        self.scale = scale if scale else 0.05 + np.random.random_sample() / 20
        self.rotation = np.deg2rad(rotation) if rotation else np.deg2rad(np.random.random_sample() * 360)
        PlanarSource.__init__(self, **kwargs)

    def define_radiation_shape(self, field_shape):
        x, y = make_grid(field_shape)
        r, phi = cartesian_to_polar((x - self.coordinates[Coordinate.X]) * x.shape[Coordinate.Y] / np.max(x.shape),
                                    (y - self.coordinates[Coordinate.Y]) * x.shape[Coordinate.X] / np.max(x.shape))

        n = int(np.floor(zero_division(self.n_vertex, np.modf(self.n_vertex)[0])) + 1)
        res = np.zeros(field_shape).T

        for i in range(n):
            res += np.heaviside((self.scale / np.cos(2 * np.pi / self.n_vertex *
                                                     (1 / 2 * np.floor(self.n_vertex *
                                                                       (phi + self.rotation + 2 * np.pi * i) / np.pi)
                                                      - np.floor(1 / 2 * np.floor(self.n_vertex *
                                                                                  (phi + self.rotation + 2 * np.pi * i)
                                                                                  / np.pi)))
                                                     - (phi + self.rotation + 2 * np.pi * i)
                                                     + np.pi / self.n_vertex
                                                     * np.floor(self.n_vertex *
                                                                (phi + self.rotation + 2 * np.pi * i) / np.pi)
                                                     ) - r), 0)
        self.radiation_shape = np.heaviside(res, 0)


class SourcePlane:
    Source = {
        'point': PointSource,
        'circular': CircularSource,
        'rectangular': RectangularSource,
        'triangular': TriangularSource,
        'rotational': RotationalFigureSource,
        'polygonal': PolygonalSource
    }

    SourceTypeSelector = ('point', 'circular', 'rectangular', 'triangular', 'rotational', 'polygonal')

    def __init__(self, shape, sources=np.array(list())):
        self.shape = shape
        self.sources = sources

    def add_source(self, source_type=None, **kwargs):
        if not source_type:
            source_type = self.SourceTypeSelector[np.random.randint(0, 6)]
        self.sources = np.append(self.sources, [self.Source[source_type](field_shape=self.shape, **kwargs)])

    def show_sources_positions(self):
        res = np.zeros(self.shape).T
        for source in self.sources:
            res += source.radiation_shape
        return res

    def show_radiation(self):
        res = np.zeros(self.shape).T
        for source in self.sources:
            res += source.radiate()
        return res


def make_grid(shape):
    return np.meshgrid(np.arange(shape[Coordinate.X]) / shape[Coordinate.X],
                       np.arange(shape[Coordinate.Y]) / shape[Coordinate.Y])


def calculate_line(points):
    return lambda p: (p[Coordinate.X] * (points[0][Coordinate.Y] - points[1][Coordinate.Y])
                      + p[Coordinate.Y] * (points[1][Coordinate.X] - points[0][Coordinate.X])
                      + (points[0][Coordinate.X] * points[1][Coordinate.Y] -
                         points[0][Coordinate.Y] * points[1][Coordinate.X]))
