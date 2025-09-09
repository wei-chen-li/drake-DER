import numpy as np
import pyvista as pv
from pydrake.all import Filament, RigidTransform, RotationMatrix


def MakeFilamentMesh(filament, circumferential_resolution):
    assert filament.closed()

    node_pos = filament.node_pos()
    edge_m1 = filament.edge_m1()
    num_nodes = node_pos.shape[1]
    num_edges = edge_m1.shape[1]

    edge_t = []
    for i in range(num_edges):
        ip1 = (i + 1) % num_nodes
        t = node_pos[:, ip1] - node_pos[:, i]
        t /= np.linalg.norm(t)
        edge_t.append(t)
    edge_t = np.array(edge_t).T

    p_CVs, triangles = MakeCrossSectionMesh(
        filament.cross_section(), circumferential_resolution)

    verts = []
    elems = []
    for i in range(num_nodes):
        pos = node_pos[:, i]
        im1 = (i - 1 + num_edges) % num_edges
        t = (edge_t[:, i] + edge_t[:, im1]) / 2
        m1 = (edge_m1[:, i] + edge_m1[:, im1]) / 2
        m1 -= m1.dot(t) * t
        m1 /= np.linalg.norm(m1)
        R_WC = np.vstack([m1, np.cross(t, m1), t]).T
        X_WC = RigidTransform(RotationMatrix.MakeUnchecked(R_WC), pos)
        verts.append(X_WC @ p_CVs)
    for i in range(num_nodes):
        offset0 = p_CVs.shape[1] * i
        offset1 = p_CVs.shape[1] * ((i + 1) % num_nodes)
        for k in range(triangles.shape[1]):
            tri = triangles[:, k]
            elems.extend(SplitTriangularPrismToTetrahedra(
                offset0 + tri[0], offset0 + tri[1], offset0 + tri[2],
                offset1 + tri[0], offset1 + tri[1], offset1 + tri[2]))
    verts = np.hstack(verts, dtype=float)
    elems = np.array(elems, dtype=int).T
    return verts, elems


def MakeCrossSectionMesh(cross_section, resolution):
    assert isinstance(cross_section, Filament.RectangularCrossSection)
    w, h = cross_section.width, cross_section.height
    num_w_div = max(round(w / resolution), 1)
    num_h_div = max(round(h / resolution), 1)

    p_CVs = []
    triangles = []

    for i in range(num_w_div + 1):
        for j in range(num_h_div + 1):
            p_CVs.append([
                w / num_w_div * i - w / 2,
                h / num_h_div * j - h / 2,
                0.0])
    for i in range(num_w_div):
        for j in range(num_h_div):
            idx1 = (num_h_div + 1) * i + j
            idx2 = idx1 + 1
            idx3 = (num_h_div + 1) * (i + 1) + j
            idx4 = idx3 + 1
            triangles.append([idx1, idx3, idx4])
            triangles.append([idx1, idx4, idx2])

    p_CVs = np.array(p_CVs, dtype=float).T
    triangles = np.array(triangles, dtype=int).T
    return p_CVs, triangles


def SplitTriangularPrismToTetrahedra(v0, v1, v2, v3, v4, v5):
    return [
        (v0, v1, v2, v3),
        (v1, v2, v3, v4),
        (v2, v3, v4, v5)
    ]


def WriteVtk(filename, verts, tets):
    """
    verts: (3,N) array
    tets: (4,M) array of vertex indices
    """
    points = verts.T  # (N,3)

    cells = []
    for tet in tets.T:
        cells.append(4)
        cells.extend(tet)
    cells = np.array(cells)

    # Cell type 10 = VTK_TETRA
    celltypes = np.array([10] * tets.shape[1])

    # Build UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    # Write to .vtk file
    grid.save(filename, binary=False)
    print(f"Mesh saved to {filename}")
