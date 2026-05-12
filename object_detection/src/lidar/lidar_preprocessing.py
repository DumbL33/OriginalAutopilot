import math
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import linear_sum_assignment

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class LidarPreprocessing:
    def __init__(self):
        self.start_idx = None
        self.clusters = {}
        self.matched_clusters = {}
        self.msg = None
        self.previous_cluster = {}
        self.previous_ranges = []
        self.filter_ranges = []
        print("Start preprocessing!")

    def filter_data(self, msg):
        self.msg = msg

        center_angle = math.radians(180)
        half_fov = math.radians(30)

        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(len(msg.ranges))])
        angles = np.mod(angles + 2 * np.pi, 2 * np.pi)
        center = np.mod(center_angle, 2 * np.pi)
        lower = np.mod(center - half_fov, 2 * np.pi)
        upper = np.mod(center + half_fov, 2 * np.pi)

        if lower < upper:
            mask = (angles >= lower) & (angles <= upper)
        else:

            mask = (angles >= lower) | (angles <= upper)

        self.filter_ranges = list(np.array(msg.ranges)[mask])
        self._used_indices = np.where(mask)[0]

        if not self.previous_ranges:
            self.previous_ranges = self.filter_ranges
        for i in range(len(self.filter_ranges)):
            if np.isnan(self.filter_ranges[i]) or np.isinf(self.filter_ranges[i]) or self.filter_ranges[i] < self.msg.range_min or self.filter_ranges[i] > self.msg.range_max:
                if len(self.filter_ranges) >= 2 and i <= len(self.filter_ranges) - 1:
                    correct_next_neighbor = []
                    k = i
                    while len(correct_next_neighbor) < 3 and k < min(len(self.filter_ranges), i + 3):
                        next_neighbor = self.filter_ranges[k]
                        if not np.isinf(next_neighbor) and self.msg.range_min <= next_neighbor <= self.msg.range_max:
                            correct_next_neighbor.append(next_neighbor)
                        k += 1
                        if k>=i + 3 and len(correct_next_neighbor)==0:
                            index=k
                            while index<len(self.filter_ranges):
                                if not np.isinf(self.filter_ranges[index]) and self.msg.range_min <= self.filter_ranges[index] <= self.msg.range_max:
                                    correct_next_neighbor.append(self.filter_ranges[index])
                                    break
                                index+=1
                    neighbors = []
                    if i >= 2:
                        neighbors.append(self.filter_ranges[i - 2])
                    if i >= 1:
                        neighbors.append(self.filter_ranges[i - 1])

                    neighbors += correct_next_neighbor

                    if len(neighbors) == 0:
                        self.filter_ranges[i] = self.filter_ranges[i - 1]
                    else:
                        new_value = np.median(neighbors)
                        if i!=0 and abs(new_value - self.filter_ranges[i - 1]) > 0.5:
                            self.filter_ranges[i] = self.filter_ranges[i - 1]
                        else:
                            self.filter_ranges[i] = new_value
                else:
                    self.filter_ranges[i] = self.msg.range_max
            if len(self.previous_ranges) == len(self.filter_ranges):
                if i < len(self.previous_ranges) and abs(self.filter_ranges[i] - self.previous_ranges[i]) < 0.05:
                    self.filter_ranges[i] = self.previous_ranges[i]
            else:
                print(f"[ERROR]: Length mismatch! previous: {len(self.previous_ranges)}, current: {len(self.filter_ranges)}")
        self.previous_ranges = self.filter_ranges.copy()
        # print("[INFO]: Filtering complete")
        return self.filter_ranges

    def __point_conversion(self):
        filter_ranges = np.array(self.filter_ranges)
        indices = self._used_indices
        angles = self.msg.angle_min + indices * self.msg.angle_increment

        x = filter_ranges * np.cos(angles)
        y = filter_ranges * np.sin(angles)

        mask = (filter_ranges > self.msg.range_min) & (filter_ranges < self.msg.range_max)

        # Поворачиваем на 180 градусов (инвертируем координаты)
        x_rot = -x[mask]  # инвертируем ось x
        y_rot = -y[mask]  # инвертируем ось y

        return np.column_stack((x_rot, y_rot))

    def __match_cluster(self):
        matched_clusters = {}

        if not self.previous_cluster:
            self.previous_cluster = self.clusters
            return self.clusters

        prev_keys = list(self.previous_cluster.keys())
        current_keys = list(self.clusters.keys())

        cost_matrix = np.zeros((len(prev_keys), len(current_keys)))

        for i, prev_key in enumerate(prev_keys):
            for j, curr_key in enumerate(current_keys):
                cost_matrix[i, j] = np.linalg.norm(
                    np.mean(self.previous_cluster[prev_key], axis=0) - np.mean(self.clusters[curr_key], axis=0)
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        threshold = 0.1
        alpha = 0.6

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < threshold:
                prev_centroid = np.mean(self.previous_cluster[prev_keys[i]], axis=0)
                curr_centroid = np.mean(self.clusters[current_keys[j]], axis=0)
                new_centroid = alpha * prev_centroid + (1 - alpha) * curr_centroid
                offset = new_centroid - curr_centroid
                matched_clusters[prev_keys[i]] = self.clusters[current_keys[j]] + offset
            else:
                matched_clusters[current_keys[j]] = self.clusters[current_keys[j]]

        for key in self.clusters.keys():
            if key not in matched_clusters:
                matched_clusters[key] = self.clusters[key]

        self.previous_cluster = matched_clusters
        return matched_clusters

    def __smooth_points(self, points, window_size=3):
        if len(points) < window_size:
            return points

        smoothed_points = np.copy(points)
        for i in range(len(points)):
            start = max(0, i - window_size // 2)
            end = min(len(points), i + window_size // 2 + 1)
            if len(points[start:end]) >= window_size // 2:
                smoothed_points[i] = np.mean(points[start:end], axis=0)

        return smoothed_points

    def apply_dbscan(self):
        points = self.__point_conversion()

        if len(points) == 0:
            print("[ERROR]: No valid points for clustering")
            return

        dbscan = DBSCAN(eps=0.13, min_samples=5)
        labels = dbscan.fit_predict(points)

        self.clusters = {}
        for label in set(labels):
            if label != -1:
                self.clusters[label] = points[labels == label]

        self.matched_clusters = self.__match_cluster()


        # self.__visualize_clusters(points, labels, self.matched_clusters)


    def __visualize_clusters(self, points, labels, matched_clusters):
        plt.clf()
        unique_labels = set(labels)

        num_noise = np.sum(labels == -1)

        for label in unique_labels:
            if label == -1:
                plt.scatter(points[labels == label][:, 0], points[labels == label][:, 1], c='gray', s=10, label='Noisy')
            elif label in matched_clusters:
                plt.scatter(matched_clusters[label][:, 0], matched_clusters[label][:, 1], label=f'Cluster {label}')
            else:
                print(f"[ERROR]: Unexpected cluster label: {label}")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('DBSCAN LiDAR Clustering')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.draw()
        plt.pause(0.01)

    def get_all_clusters_points(self):
        obstacles = []
        for cluster_id, points in self.matched_clusters.items():
            if len(points) < 3:
                continue
            cluster = []
            for point in points:
                cluster.append(tuple(point))
            obstacles.append(cluster)
        return obstacles


    def find_optimal_eps(self, points, min_samples=5):
        neigh = NearestNeighbors(n_neighbors=min_samples)
        neigh.fit(points)
        distances, _ = neigh.kneighbors(points)
        distances = np.sort(distances[:, -1])
        plt.figure(figsize=(8, 6))
        plt.plot(distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
        plt.title("Optimal eps Selection")
        plt.grid()
        plt.show()
