import numpy as np
from argparse import ArgumentParser, Namespace
import cv2
import ctypes
from ctypes import pointer, util
from lucid.misc.gl.glcontext import create_opengl_context
import OpenGL.GL as gl
from adaptmesh import triangulate

def MVC_Clone(src, tar, mask, pos_x, pos_y):
    src = src.astype(np.float64)
    src_h, src_w, _ = src.shape
    tar = tar.astype(np.float64)
    tar_h, tar_w, _ = tar.shape
    
    # mvc
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary_pts = contours[0].reshape(-1, 2)
    mask_in = mask.copy()
    mask_in[boundary_pts[:, 1], boundary_pts[:, 0]] = 0
    contours, hierarchy = cv2.findContours(mask_in.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mesh = triangulate(contours[0].reshape(-1, 2), quality=0.95) 
    mesh_pts = np.array(list(zip(mesh.p[0], mesh.p[1])))
    lambda_ = np.zeros((len(mesh_pts), len(boundary_pts) - 1))
    for i, mesh_pt in enumerate(mesh_pts):
        tangent = np.array([0] * len(boundary_pts), dtype=np.float64)
        l0 = (boundary_pts[0:-1, :] - mesh_pt).astype(np.float64)
        l1 = (boundary_pts[1:, :] - mesh_pt).astype(np.float64)
        dot_ = np.sum(l0*l1, 1)
        cos_ = dot_/np.sqrt(np.sum(l0*l0, 1))/np.sqrt(np.sum(l1*l1, 1))
        theta = np.arccos(np.clip(0, cos_, 1)) 
        tangent[0] = np.tan(theta[-1] / 2)
        tangent[1:] = np.tan(theta/2)
        weight = (tangent[:-1]+tangent[1:])/ np.sqrt(np.sum((boundary_pts[0:-1, :]-mesh_pt) * (boundary_pts[0:-1, :]-mesh_pt),1))
        weight = weight / np.sum(weight)
        lambda_[i, :] = weight

    # boundary difference
    img_pos = boundary_pts[:-1] + (pos_x, pos_y)
    in_img = ((img_pos[:, 0] >= 0) & (img_pos[:, 1] >= 0) & (img_pos[:, 0] < tar_w) & (img_pos[:, 1] < tar_h))
    x, y = boundary_pts[:-1, 0][in_img], boundary_pts[:-1, 1][in_img]
    diff = tar[y + pos_y, x + pos_x, :] - src[y, x, :]

    # mean-value interpolant
    lambda_in = lambda_[:, in_img]
    lambda_in = lambda_in / np.sum(lambda_in, axis=1).reshape(-1, 1)
    r = lambda_in @ diff
    mx, mn = r.max(), r.min()
    create_opengl_context((tar_w, tar_h))
    gl.glClearColor(1, 1, 1, 0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    mesh_pts = np.array(list(zip(mesh.p[0], mesh.p[1])))
    tris = np.array(list(zip(mesh.t[0], mesh.t[1], mesh.t[2])))
    mesh_pts = mesh_pts + (pos_x, pos_y)
    r = (r - mn) / (mx - mn)

    gl.glBegin(gl.GL_TRIANGLES)
    for tri in tris:
        gl.glColor3f(r[tri][0][0], r[tri][0][1], r[tri][0][2])
        gl.glVertex3f(mesh_pts[tri][0][0]*(2/tar_w)-1,mesh_pts[tri][0][1]*(-2/tar_h)+1, 0)
        gl.glColor3f(r[tri][1][0], r[tri][1][1], r[tri][1][2])
        gl.glVertex3f(mesh_pts[tri][1][0]*(2/tar_w)-1,mesh_pts[tri][1][1]*(-2/tar_h)+1, 0)
        gl.glColor3f(r[tri][2][0], r[tri][2][1], r[tri][2][2])
        gl.glVertex3f(mesh_pts[tri][2][0]*(2/tar_w)-1,mesh_pts[tri][2][1]*(-2/tar_h)+1, 0)
    gl.glEnd()
    
    buf = gl.glReadPixelsub(0, 0, tar_w, tar_h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    r_x = np.frombuffer(buf, dtype=np.uint8).reshape(tar_h, tar_w, 3)[::-1]
    r_x = (r_x/255).astype(np.float32)
    r_x = r_x * (mx - mn) + mn
    src_pts = (r_x != r_x.max()).all(axis=2)
    src_pts = np.argwhere(src_pts)
    src_pts_in = src_pts - [pos_y, pos_x]
    in_src = src_pts[((src_pts_in[:, 0] >= 0) & (src_pts_in[:, 1] >= 0) & (src_pts_in[:, 0] < src_h) & (src_pts_in[:, 1] < src_w))]
    y, x = in_src[:, 0], in_src[:, 1]
    tar[y, x] = src[y - pos_y, x - pos_x] + r_x[y, x]
    tar = np.clip(0, tar, 255).astype(np.uint8)

    return tar