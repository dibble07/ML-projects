# Import libraries
import numpy as np
import PySimpleGUI as sg
import random

# Define user functions

def UserMoveInput(board_in, player):
	print(board_in)
	pos_acc=False
	while pos_acc == False:
		pos_str=sg.popup_get_text("Player {0}, make your move. [1-9]".format(player), 'Input')
		try:
			pos_out=int(pos_str)
			if pos_out>=1 and pos_out<=9:
				pos_acc=True
			else:
				print("Input must be an integer between 1 and 9")
		except:
			print("Input must be an integer")
	pos_out-=1
	return pos_out

def RandMoveInput():
	col_out = np.random.randint(0, high=9)
	return col_out

def RandSmartMoveInput(board_in, player):
	win_pos={1:[], 2:[]}
	opposition = [i for i in win_pos.keys() if i != player][0]
	for pos in range(9):
		# identify winning locations
		board_copy = board_in.copy()
		board_copy, flag = InputMove(board_copy, pos, player)
		if flag == 1:
			win, __ = BoardState(board_copy, player)
			if win:
				win_pos[player].append(pos)
		# identify win blocking locations
		board_copy = board_in.copy()
		board_copy, flag = InputMove(board_copy, pos, opposition)
		if flag == 1:
			win, __ = BoardState(board_copy, opposition)
			if win:
				win_pos[opposition].append(pos)
	if len(win_pos[player]) > 0:
		# if can win
		pos_out=min(win_pos[player])
	elif len(win_pos[opposition]) > 0:
		# if can't win but needs to block opposition win
		pos_out=min(win_pos[opposition])
	else:
		# i guess we'll choose a random empty one
		pos_out = random.choice(np.argwhere(board_in.reshape(-1) == 0))[0]
	return pos_out

def NeuralNetworkMoveInput(board_in, model):
	play_1_bool = np.zeros(board_in.shape, dtype=int)
	play_1_bool[board_in == 1] = 1
	play_2_bool = np.zeros(board_in.shape, dtype=int)
	play_2_bool[board_in == 2] = 1
	play_all_bool=np.dstack((play_1_bool,play_2_bool))
	play_all_bool=play_all_bool.reshape(-1)
	col_out_prob = model.activate(play_all_bool)
	col_out = np.argmax(col_out_prob)
	return col_out

def BoardState(board_in, player_in):
	# is the board full
	full_out=(board_in != 0).all()
	# has anyone won
	board_sz = board_in.shape
	winning_out=False
	# if enough moves have been made for someone to win then check whether player_in has won
	if len(np.nonzero(board_in.reshape(-1))[0]) >= 4:
		# check horizontal direction
		for i_row in range(board_sz[0]):
			if all(np.equal(board_in[i_row,:], player_in)):
				winning_out=True
		# check vertical direction
		for i_col in range(board_sz[0]):
			if all(np.equal(board_in[:,i_col], player_in)):
				winning_out=True
		# check southeast diagonal direction
		if all(np.equal(np.diagonal(board_in), player_in)):
			winning_out=True
		# check southwest diagonal direction
		board_in_fl=np.flip(board_in, axis=0)
		if all(np.equal(np.diagonal(board_in_fl), player_in)):
			winning_out=True
	return winning_out, full_out

def InputMove(board_in, pos_in, player):
	board_flat = board_in.reshape(-1)
	pos_emp=board_flat[pos_in] == 0
	if pos_emp:
		board_flat[pos_in]=player
		flag_out=1
	else:
		flag_out=0
	board_out = board_flat.reshape(board_in.shape)	
	return board_out, flag_out

def PlayGame(user_type_in, model, play_start):
	board_curr = np.zeros((3,3), dtype=int)
	board_out = np.copy(board_curr)
	move_out = []
	cont=True
	play_curr=play_start
	if play_curr == 1:
		opp_curr=2
	elif play_curr == 2:
		opp_curr=1
	while cont:
		# make move
		if user_type_in[play_curr] == 'User':
			pos_des = UserMoveInput(board_curr, play_curr)
		elif user_type_in[play_curr] == 'Rand':
			pos_des = RandMoveInput()
		elif user_type_in[play_curr] == 'RandSmart':
			pos_des = RandSmartMoveInput(board_curr, play_curr)
		elif user_type_in[play_curr] == 'NeuralNetwork':
			pos_des = NeuralNetworkMoveInput(board_curr, model)
		else:
			print("No player type selected")
		board_curr, flag = InputMove(board_curr, pos_des, play_curr)
		board_out = np.dstack((board_out,board_curr))
		move_out.append((play_curr, pos_des))
		if flag == 0:
			# if move out of bounds
			cont=False
			winner=-play_curr
		elif flag == 1:
			win, full = BoardState(board_curr, play_curr)
			if win == True:
				cont=False
				winner=play_curr
			elif full:
				cont=False
				winner=0
			else:
				play_curr, opp_curr = opp_curr, play_curr
	if user_type_in[play_curr] == 'User' or user_type_in[opp_curr] == 'User':
		print(board_curr)
	return board_out, move_out, winner