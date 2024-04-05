import numpy as np

def build_percentiles(error, error_thresholds, object_diameter):
  per_ = []
  for i in error_thresholds:
    items_below_threshold = np.count_nonzero(error[error<i])
    per_.append(items_below_threshold/len(error))

  items_below_threshold = np.count_nonzero(error[error<0.1*object_diameter])

  return np.array(per_),items_below_threshold / len(error)

def transform_pts_Rt(pts, R, t):
  """Applies a rigid transformation to 3D points.

  :param pts: nx3 ndarray with 3D points.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx3 ndarray with transformed 3D points.
  """
  assert (pts.shape[1] == 3)
  pts_t = R.dot(pts.T) + t.reshape((3, 1))
  return pts_t.T

def add_epos(R_est, t_est, R_gt, t_gt, pts):
  """Average Distance of Model Points for objects with no indistinguishable
  views - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = transform_pts_Rt(pts, R_est, t_est)
  pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
  e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
  return e

def ADD(modelPoints,Rgt,Rest,Tgt,Test):

    gts = Rgt[None,:].dot(modelPoints.T) + Tgt.reshape((3,1))
    ests = Rest[None,:].dot(modelPoints.T) + Test.reshape((3,1))
    add = np.linalg.norm(ests.T-gts.T,axis=1).mean()
   
    return add

def get_percentiles(metric_data,percentiles):
   
   perc = np.percentile(metric_data,percentiles)
   str_out = f"Percenteage of poses under:\t\tADD\n"
   for p in range(len(percentiles)):
    str_out+= f"{percentiles[p]}\t\t\t\t\t{perc[p]}\n"

   print(str_out)
   return np.percentile(metric_data,percentiles)