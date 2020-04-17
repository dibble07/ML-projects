# Import libraries
import numpy as np
import PySimpleGUI as sg
import random

# Define user functions

def UserMoveInput(board_in, player):
	print(board_in)
	col_acc=False
	while col_acc == False:
		col_str=sg.popup_get_text(f"Player {player}, make your move. [1-7]", 'Input')
		try:
			col_out=int(col_str)
			if col_out>=1 and col_out<=7:
				col_acc=True
			else:
				print("Input must be an integer between 1 and 7")
		except:
			print("Input must be an integer")
	col_out-=1
	return col_out

def RandMoveInput():
	col_out = np.random.randint(0, high=7)
	return col_out

def RandSmartMoveInput(board_in, player):
	win_col={1:[], 2:[]}
	opposition = [i for i in win_col.keys() if i != player][0]
	for col in range(7):
		# identify winning locations
		board_copy = board_in.copy()
		board_copy, flag = InputMove(board_copy, col, player)
		if flag == 1:
			win, __ = BoardState(board_copy)
			if win[player]:
				win_col[player].append(col)
		# identify win blocking locations
		board_copy = board_in.copy()
		board_copy, flag = InputMove(board_copy, col, opposition)
		if flag == 1:
			win, __ = BoardState(board_copy)
			if win[opposition]:
				win_col[opposition].append(col)
	if len(win_col[player]) > 0:
		# if can win
		col_out=min(win_col[player])
	elif len(win_col[opposition]) > 0:
		# if can't win but needs to block opposition win
		col_out=min(win_col[opposition])
	else:
		# i guess we'll choose a random empty one
		col_out = random.choice(np.argwhere(board_in[0,:] == 0))[0]
	return col_out

def NeuralNetworkMoveInput(board_in, model):
	play_1_bool = np.zeros(board_in.shape, dtype=int)
	play_1_bool[board_in == 1] = 1
	play_2_bool = np.zeros(board_in.shape, dtype=int)
	play_2_bool[board_in == 2] = 1
	play_all_bool=np.dstack((play_1_bool,play_2_bool))
	play_all_bool=play_all_bool.reshape(1,play_all_bool.shape[0],play_all_bool.shape[1],play_all_bool.shape[2])
	col_out_prob = model.predict(play_all_bool)
	col_out = np.argmax(col_out_prob)
	# board_flat = board_in.reshape(-1,board_in.shape[0]*board_in.shape[1])
	# col_out_prob = model.predict(board_flat)
	return col_out

def BoardState(board_in):
	# is the board full1
	full_out=all(board_in[0,:] != 0)
	# has anyone won
	board_sz = board_in.shape
	winning_out={1: False, 2: False}
	for cand in [1,2]:
		for i_row in range(board_sz[0]):
			for i_col in range(board_sz[1]):
				if i_row <= board_sz[0]-4:
					# check vertical direction
					if all(np.equal(board_in[i_row:i_row+4,i_col], cand)):
						winning_out[cand]=True
				if i_col <= board_sz[1]-4:
					# check horizontal direction
					if all(np.equal(board_in[i_row,i_col:i_col+4], cand)):
						winning_out[cand]=True
				if i_row <= board_sz[0]-4 and  i_col <= board_sz[1]-4:
					mat=board_in[i_row:i_row+4,i_col:i_col+4]
					mat_fl=np.flip(mat, axis=0)
					# check swinning_outheast diagonal direction
					if all(np.equal(np.diagonal(mat), cand)):
						winning_out[cand]=True
					# check swinning_outhwest diagonal direction
					if all(np.equal(np.diagonal(mat_fl), cand)):
						winning_out[cand]=True		
	return winning_out, full_out

def InputMove(board_in, col, player):
	row_emp=np.argwhere(board_in[:,col] == 0)
	if len(row_emp) == 0:
		flag_out=0
	else:
		row_bot=max(row_emp)
		board_in[row_bot,col]=player
		flag_out=1
	return board_in, flag_out

def PlayGame(user_type_in, model, play_start):
	board_out = np.zeros((6,7), dtype=int)
	cont=True
	play_curr=play_start
	if play_curr == 1:
		opp_curr=2
	elif play_curr == 2:
		opp_curr=1
	while cont:
		# make move
		if user_type_in[play_curr] == 'User':
			col_des = UserMoveInput(board_out, play_curr)
		elif user_type_in[play_curr] == 'Rand':
			col_des = RandMoveInput()
		elif user_type_in[play_curr] == 'RandSmart':
			col_des = RandSmartMoveInput(board_out, play_curr)
		elif user_type_in[play_curr] == 'NeuralNetwork':
			col_des = NeuralNetworkMoveInput(board_out, model)
		else:
			print("No player type selected")
		board_out, flag = InputMove(board_out, col_des, play_curr)
		if flag == 0:
			# if move out of bounds
			cont=False
			result=opp_curr
		elif flag == 1:
			win, full = BoardState(board_out)
			if win[1] == True and win[2] == False:
				cont=False
				result=1
			elif win[1] == False and win[2] == True:
				cont=False
				result=2
			elif full:
				cont=False
				result=0
			else:
				play_curr, opp_curr = opp_curr, play_curr
	return board_out, result