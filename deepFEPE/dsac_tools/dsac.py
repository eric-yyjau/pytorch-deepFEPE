import torch
import torch.nn.functional as F
import random

# import os
# import sys
# sys.path.append(os.getcwd())
# from .utils_F import *
import dsac_tools.utils_F as utils_F
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class DSAC:
	'''
	Differentiable RANSAC to robustly fit lines.
	'''

	def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, K, loss_function):
		'''
		Constructor.

		hyps -- number of line hypotheses sampled for each image
		inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
		inlier_beta -- scaling factor within the sigmoid of the soft inlier count
		inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
		loss_function -- function to compute the quality of estimated line parameters wrt ground truth
		'''

		self.hyps = hyps
		self.inlier_thresh = inlier_thresh
		self.inlier_beta = inlier_beta
		self.inlier_alpha = inlier_alpha
		self.loss_function = loss_function
		self.K = K

	def __sample_hyp(self, X, Y):
		'''
		Calculate an H hypothesis from 4 random correspondences.

		X: [N, 2]
		Y: [N, 2]
		'''

		# select 4 random correspomndences
		idx = random.sample(range(self.N), 10)
		# idx = [0, 1, 2, 3]
		# print(idx)
		X_sample = X[idx, :]
		Y_sample = Y[idx, :]

		return utils_F._E_from_XY(X_sample, Y_sample, self.K), idx
		# Es =  utils_F._E_from_XY_batch(X_sample.unsqueeze(0).cuda(), Y_sample.unsqueeze(0).cuda(), self.K.unsqueeze(0).cuda())
		return Es.cpu().squeeze(0), idx

	def __soft_inlier_count(self, H, X, Y):
		'''
		Soft inlier count for a given line and a given set of points.

		slope -- slope of the line
		intercept -- intercept of the line
		x -- vector of x values
		y -- vector of y values
		'''

		# point line distances
		dists_sampson = utils_F._sampson_dist(utils_F.E_to_F(H, self.K), X, Y, if_homo=False)
		# print(dists.detach().numpy())
		# print(np.max(dists.detach().numpy()))

		# soft inliers
		dists = 1 - torch.sigmoid(self.inlier_beta * (dists_sampson - self.inlier_thresh))
		# print(dists.detach().numpy())
		score = torch.sum(dists)

		return score, dists, dists_sampson

	def __refine_hyp(self, X, Y, Ws):
		'''
		Refinement by weighted Deming regression.

		Fits a line minimizing errors in x and y, implementation according to:
			'Performance of Deming regression analysis in case of misspecified
			analytical error ratio in method comparison studies'
			Kristian Linnet, in Clinical Chemistry, 1998

		x -- vector of x values
		y -- vector of y values
		weights -- vector of weights (1 per point)
		'''

		return utils_F._E_from_XY(X, Y, self.K, torch.diag(Ws))

	def __call__(self, X, Y, H_gt):
		'''
		Perform robust, differentiable line fitting according to DSAC.

		Returns the expected loss of choosing a good line hypothesis which can be used for backprob.

		prediction -- predicted 2D points for a batch of images, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of point dimensions (y, x)
		labels -- ground truth labels for the batch, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of parameters (intercept, slope)
		'''
		
		print('>>>>>>>>>>>>>>>> Running DSAC... ---------------')

		# working on CPU because of many, small matrices
		X = X.cpu()
		Y = Y.cpu()
		H_gt = H_gt.cpu()

		self.N = X.size(0)
		assert X.size(0) == Y.size(0), 'N mismatch between X and Y!'

		avg_exp_loss = 0 # expected loss
		avg_top_loss = 0 # loss of best hypothesis

		# self.est_parameters = torch.zeros(batch_size, 2) # estimated lines
		self.est_losses = torch.zeros(1) # loss of estimated lines
		self.inliers = torch.zeros(self.N) # (soft) inliers for estimated lines

		hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
		hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis
		self.inlier_scores = torch.zeros([self.hyps, self.N]) # score of each corre. w.r.t each hypo
		self.sampson_dists = torch.zeros([self.hyps, self.N]) # Sampson distance of each corre. w.r.t each hypo

		self.max_score = 0 	# score of best hypothesis

		# y = prediction[b, 0] # all y-values of the prediction
		# x = prediction[b, 1] # all x.values of the prediction

		N_scores = torch.zeros([self.N, 1])
		N_counts = torch.zeros([self.N, 1])

		for h in range(self.hyps):

			# === step 1: sample hypothesis ===========================
			H, idx = self.__sample_hyp(X, Y)

			# === step 2: score hypothesis using soft inlier count ====
			score, dists, dists_sampson = self.__soft_inlier_count(H, X, Y)
			self.inlier_scores[h, :] = dists
			self.sampson_dists[h, :] = dists_sampson

			# === step 3: refine hypothesis ===========================
			# print(torch.sqrt(dists).numpy())
			H = self.__refine_hyp(X, Y, torch.sqrt(dists))

			# inlier_mask = dists_sampson<self.inlier_thresh
			# if torch.sum(inlier_mask).numpy() > 10:
			# 	H = utils_F._F_from_XY(X[inlier_mask, :], Y[inlier_mask, :])

			# === step 4: calculate loss of hypothesis ================
			loss = self.loss_function(H, X, Y)

			# store results
			hyp_losses[h] = loss
			hyp_scores[h] = score

			N_scores[idx] += score
			# print(N_scores)
			N_counts[idx] += 1.

			# keep track of best hypothesis so far
			if score > self.max_score:
				self.max_score = score
				# self.est_losses[b] = loss
				# self.est_parameters[b] = hyp
				self.best_H = H
				self.best_H_idx = h
				self.best_dists = dists
				self.best_corres_idx = idx
				# self.batch_inliers[b] = inliers

		# # === step 5: calculate the expectation ===========================

		# #softmax distribution from hypotheses scores
		# hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

		# # expectation of loss
		# exp_loss = torch.sum(hyp_losses * hyp_scores)
		# avg_exp_loss = avg_exp_loss + exp_loss

		# # loss of best hypothesis (for evaluation)
		# avg_top_loss = avg_top_loss + self.est_losses[b]

		# return avg_exp_loss / batch_size, avg_top_loss / batch_size

		print('<<<<<<<<<<<<<<<< DONE Running DSAC. ---------------')

		return N_scores / (N_counts+1e-10)
