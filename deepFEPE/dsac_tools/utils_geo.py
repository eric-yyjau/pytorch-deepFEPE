import torch
import numpy as np
import cv2
import math
import deepFEPE.dsac_tools.utils_misc as utils_misc

def rot_to_angle(R):
    return rot12_to_angle_error(np.eye(3, R))

# def _R_to_q(R):
#     if R[2, 2] < 0:
#         if R[0, 0] > R[1, 1]:
#             t = 1. + R[0, 0] - R[1, 1] - R[2, 2]
#             q = torch.tensor([t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2], R[1, 2]-R[2, 1]], device=R.device)
#         else:
#             t = 1. - R[0, 0] + R[1, 1] - R[2, 2]
#             q = torch.tensor([R[0, 1]+R[1, 0], t, R[1, 2]+R[2, 1], R[2, 0]-R[0, 2]], device=R.device)
#     else:
#         if R[0, 0] < -R[1, 1]:
#             t = 1. - R[0, 0] - R[1, 1] + R[2, 2]
#             q = torch.tensor([R[2, 0]+R[0, 2], R[1, 2]+R[2, 1], t, R[0, 1]-R[1, 0]], device=R.device)
#         else:
#             t = 1. + R[0, 0] + R[1, 1] + R[2, 2]
#             q = torch.tensor([R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0], t], device=R.device)
#     q *= 0.5 / (torch.sqrt(t)+1e-10)
#     return q

def _q_matrix(q):
    """Matrix representation of quaternion for multiplication purposes.
    """
    return torch.cat((
        torch.stack((q[0], -q[1], -q[2], -q[3]), 1),
        torch.stack((q[1],  q[0], -q[3],  q[2]), 1),
        torch.stack((q[2],  q[3],  q[0], -q[1]), 1),
        torch.stack((q[3], -q[2],  q[1],  q[0]), 1)), dim=0)

def _q_bar_matrix(q):
    """Matrix representation of quaternion for multiplication purposes.
    """
    return torch.cat((
        torch.stack((q[0], -q[1], -q[2], -q[3]), 1),
        torch.stack((q[1],  q[0],  q[3], -q[2]), 1),
        torch.stack((q[2], -q[3],  q[0],  q[1]), 1),
        torch.stack((q[3],  q[2], -q[1],  q[0]), 1)), dim=0)

def _rotation_matrix(q):
    """ https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L965
    Get the 3x3 rotation matrix equivalent of the quaternion rotation.
    Returns:
        A 3x3 orthogonal rotation matrix as a 3x3 Numpy array
    Note:
        This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
    """
    q = q / (torch.norm(q)+1e-10)
    product_matrix = _q_matrix(q) @ _q_bar_matrix(q).t()
    return product_matrix[1:][:,1:]

def _R_to_q(R): # _trace_method
    """ https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L204
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Altered to work with the column vector convention instead of row vectors
    
    """
    m = R.t() # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = torch.stack((m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]))
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = torch.stack((m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]))
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = torch.stack((m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t))
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = torch.stack((t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]))

    q *= 0.5 / torch.sqrt(t)
    if q[0] < 0.:
        q = -q
    return q.unsqueeze(-1)

def R_to_q_np(matrix):
    """ https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L204
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Altered to work with the column vector convention instead of row vectors
    
    """
    m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

    q = np.array(q, dtype=np.float32)
    q *= 0.5 / np.sqrt(t)
    if q[0] < 0.:
        q = -q
    return q.reshape(-1, 1)

def q_matrix_np(q):
    """Matrix representation of quaternion for multiplication purposes.
    """
    return np.array([
        [q[0, 0], -q[1, 0], -q[2, 0], -q[3, 0]],
        [q[1, 0],  q[0, 0], -q[3, 0],  q[2, 0]],
        [q[2, 0],  q[3, 0],  q[0, 0], -q[1, 0]],
        [q[3, 0], -q[2, 0],  q[1, 0],  q[0, 0]]])

def q_bar_matrix_np(q):
    """Matrix representation of quaternion for multiplication purposes.
    """
    return np.array([
        [q[0, 0], -q[1, 0], -q[2, 0], -q[3, 0]],
        [q[1, 0],  q[0, 0],  q[3, 0], -q[2, 0]],
        [q[2, 0], -q[3, 0],  q[0, 0],  q[1, 0]],
        [q[3, 0],  q[2, 0], -q[1, 0],  q[0, 0]]])
    
def q_to_R_np(q):
    """ https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L965
    Get the 3x3 rotation matrix equivalent of the quaternion rotation.
    Returns:
        A 3x3 orthogonal rotation matrix as a 3x3 Numpy array
    Note:
        This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
    """
    q = q / (np.linalg.norm(q)+1e-10)
    product_matrix = np.dot(q_matrix_np(q), q_bar_matrix_np(q).conj().transpose())
    return product_matrix[1:][:,1:]


def rot12_to_angle_error(R0, R1):
    r, _ = cv2.Rodrigues(R0.dot(R1.T))
    rotation_error_from_identity = np.linalg.norm(r) / np.pi * 180.
    # another_way = np.rad2deg(np.arccos(np.clip((np.trace(R0 @ (R1.T)) - 1) / 2, -1., 1.)))
    # print(rotation_error_from_identity, another_way)
    return rotation_error_from_identity
    # return another_way

def _rot_angle_error(R0, R1):
    rot_error = torch.acos(torch.clamp((torch.trace(R0 @ (R1.t())) - 1) / 2, -1., 1.))
    rot_error = rot_error / np.pi * 180.
    # print(rotation_error_from_identity, another_way)
    # return rotation_error_from_identity
    return rot_error

def _l2_error(t0, t1):
    trans_error = torch.norm(t0-t1, 2)
    return trans_error

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v)) + 1e-10

def vector_angle(v1, v2):
    """ v1, v2: [N, 1] or (N)
        return: angles in degree: () """
    dot_product = dotproduct(v1, v2) / (length(v1) * length(v2) + 1e-10)
    return math.acos(np.clip(dot_product, -1., 1.)) / np.pi * 180.

def dotproducts(v1s, v2s):
    return np.sum(v1s * v2s, axis=1, keepdims=True)

def vectors_angle(v1s, v2s):
    """ v1s, v2s: [N, 3]
        return: angles in degree: [N, 1] """
    dot_v1sv2s = dotproducts(v1s, v2s)
    length_v1s = np.sqrt(dotproducts(v1s, v1s))
    length_v2s = np.sqrt(dotproducts(v2s, v2s))
    return np.arccos(dot_v1sv2s / (length_v1s * length_v2s)) / np.pi * 180.

def invert_Rt(R21, t21):
    delta_Rtij = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(np.hstack((R21, t21)))))
    R12 = delta_Rtij[:, :3]
    t12 = delta_Rtij[:, 3:4]
    return R12, t12

def qmul(q, r): # https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L13
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def quat2mat(quat): # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L112
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat