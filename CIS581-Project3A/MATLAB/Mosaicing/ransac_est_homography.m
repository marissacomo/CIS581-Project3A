% File name: ransac_est_homography.m
% Author:
% Date created:

function [H, inlier_ind] = ransac_est_homography(x1, y1, x2, y2, thresh)
% Input:
%    y1, x1, y2, x2 are the corresponding point coordinate vectors Nx1 such
%    that (y1i, x1i) matches (x2i, y2i) after a preliminary matching
%    thresh is the threshold on distance used to determine if transformed
%    points agree

% Output:
%    H is the 3x3 matrix computed in the final step of RANSAC
%    inlier_ind is the nx1 vector with indices of points in the arrays x1, y1,
%    x2, y2 that were found to be inliers

% Write Your Code Here
end