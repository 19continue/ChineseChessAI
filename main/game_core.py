import copy

init_state=[[ 5, 4, 3, 2, 1, 2, 3, 4, 5],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 6, 0, 0, 0, 0, 0, 6, 0],
            [ 7, 0, 7, 0, 7, 0, 7, 0, 7],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-7, 0,-7, 0,-7, 0,-7, 0,-7],
            [ 0,-6, 0, 0, 0, 0, 0,-6, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-5,-4,-3,-2,-1,-2,-3,-4,-5]]

def get_all_move_mapping():
    id_to_action = {}
    action_to_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']
    id_tab = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    id_to_action[id_tab] = action
                    action_to_id[action] = id_tab
                    id_tab += 1

    for action in advisor_labels:
        id_to_action[id_tab] = action
        action_to_id[action] = id_tab
        id_tab += 1

    for action in bishop_labels:
        id_to_action[id_tab] = action
        action_to_id[action] = id_tab
        id_tab += 1

    return id_to_action, action_to_id


id_to_action_mapping, action_to_id_mapping = get_all_move_mapping()

def state_change_by_move(current_state,move_action):
    new_state=copy.deepcopy(current_state)
    x,y,tox,toy=int(move_action[0]),int(move_action[1]),int(move_action[2]),int(move_action[3])
    new_state[tox,toy]=new_state[x,y]
    new_state[x,y]=0
    return new_state

# 获取当前局面下的所有合法的棋子移动动作集合
def get_all_move_action(current_state,current_player):
    move_actions=[]
    river = int(4.5 * current_player - 0.5)
    home = int(4.5 * current_player - 2.5)
    for x in range(10):
        for y in range(9):
            piece=current_state[x][y] * current_player
            if piece == 0:
                continue
            if piece == 5:
                tox=x
                for toy in range(y - 1, -1, -1):
                    camp=current_state[tox][toy] * current_player
                    if camp > 0:
                        break
                    elif camp < 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                        break
                    else:
                        action=str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                for toy in range(y + 1, 9):
                    camp = current_state[tox][toy] * current_player
                    if camp > 0:
                        break
                    elif camp < 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                        break
                    else:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                toy=y
                for tox in range(x - 1, -1, -1):
                    camp=current_state[tox][toy] * current_player
                    if camp > 0:
                        break
                    elif camp < 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                        break
                    else:
                        action=str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                for tox in range(x + 1, 10):
                    camp = current_state[tox][toy] * current_player
                    if camp > 0:
                        break
                    elif camp < 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                        break
                    else:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
            elif piece == 4:
                if x - 2 >= 0 and current_state[x-1][y] == 0:
                    tox=x-2
                    toy=y-1
                    if toy >= 0 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    toy=y+1
                    if toy < 9 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                if x + 2 < 10 and current_state[x+1][y] == 0:
                    tox = x + 2
                    toy = y - 1
                    if toy >= 0 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    toy = y + 1
                    if toy < 9 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                if y - 2 >= 0 and current_state[x][y-1] == 0:
                    toy = y - 2
                    tox = x - 1
                    if tox >= 0 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    tox = x + 1
                    if tox < 10 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                if y + 2 < 9 and current_state[x][y+1] == 0:
                    toy = y + 2
                    tox = x - 1
                    if tox >= 0 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    tox = x + 1
                    if tox < 10 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
            elif piece == 3:
                for l in (-2,2):
                    tox=x+l
                    if tox < 0 or tox > 9 or tox * current_player > river:
                        continue
                    toy=y+l
                    if 0 <= toy < 9 and current_state[x+l//2][y+l//2]==0 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    toy=y-l
                    if 0 <= toy < 9 and current_state[x+l//2][y-l//2]==0 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
            elif piece == 2:
                for l in (-1,1):
                    tox=x+l
                    if tox < 0 or tox > 9 or tox * current_player > home:
                        continue
                    toy=y+l
                    if 3<=toy<=5 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    toy=y-l
                    if 3<=toy<=5 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
            elif piece == 1:
                for step in (-1,1):
                    tox=x
                    toy=y+step
                    if 3<=toy<=5 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    toy=y
                    tox=x+step
                    if 0<=tox<10 and tox * current_player <= home and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
            elif piece == 6:
                tox = x
                interval = 0
                for toy in range(y - 1, -1, -1):
                    camp = current_state[tox][toy] * current_player
                    if camp > 0:
                        interval += 1
                        if interval > 1:
                            break
                    elif camp < 0:
                        if interval == 1:
                            action = str(x) + str(y) + str(tox) + str(toy)
                            move_actions.append(action)
                            break
                        elif interval > 1:
                            break
                        else:
                            interval += 1
                    elif interval == 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                interval = 0
                for toy in range(y + 1, 9):
                    camp = current_state[tox][toy] * current_player
                    if camp > 0:
                        interval += 1
                        if interval > 1:
                            break
                    elif camp < 0:
                        if interval == 1:
                            action = str(x) + str(y) + str(tox) + str(toy)
                            move_actions.append(action)
                            break
                        elif interval > 1:
                            break
                        else:
                            interval += 1
                    elif interval == 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                toy = y
                interval = 0
                for tox in range(x - 1, -1, -1):
                    camp = current_state[tox][toy] * current_player
                    if camp > 0:
                        interval += 1
                        if interval > 1:
                            break
                    elif camp < 0:
                        if interval == 1:
                            action = str(x) + str(y) + str(tox) + str(toy)
                            move_actions.append(action)
                            break
                        elif interval > 1:
                            break
                        else:
                            interval += 1
                    elif interval == 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                interval = 0
                for tox in range(x + 1, 10):
                    camp = current_state[tox][toy] * current_player
                    if camp > 0:
                        interval += 1
                        if interval > 1:
                            break
                    elif camp < 0:
                        if interval == 1:
                            action = str(x) + str(y) + str(tox) + str(toy)
                            move_actions.append(action)
                            break
                        elif interval > 1:
                            break
                        else:
                            interval += 1
                    elif interval == 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
            elif piece == 7:
                toy=y
                tox=x+current_player
                if 0<=tox<10 and current_state[tox][toy] * current_player <= 0:
                    action = str(x) + str(y) + str(tox) + str(toy)
                    move_actions.append(action)
                if x * current_player > river:
                    tox=x
                    toy=y+1
                    if toy<9 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
                    toy=y-1
                    if toy>=0 and current_state[tox][toy] * current_player <= 0:
                        action = str(x) + str(y) + str(tox) + str(toy)
                        move_actions.append(action)
    return move_actions

def general_warn(current_state,current_player,opponent_general_position):
    general_x=int(opponent_general_position[0])
    general_y=int(opponent_general_position[1])
    for x in range(10):
        for y in range(9):
            piece=current_state[x][y] * current_player
            if piece == 0:
                continue
            if piece == 5:
                if x == general_x:
                    tox = x
                    for toy in range(y - 1, -1, -1):
                        camp=current_state[tox][toy] * current_player
                        if camp > 0:
                            break
                        elif camp < 0:
                            if toy == general_y:
                                return True
                            else:
                                break
                    for toy in range(y + 1, 9):
                        camp = current_state[tox][toy] * current_player
                        if camp > 0:
                            break
                        elif camp < 0:
                            if toy == general_y:
                                return True
                            else:
                                break
                elif y == general_y:
                    toy = y
                    for tox in range(x - 1, -1, -1):
                        camp=current_state[tox][toy] * current_player
                        if camp > 0:
                            break
                        elif camp < 0:
                            if tox == general_x:
                                return True
                            else:
                                break
                    for tox in range(x + 1, 10):
                        camp = current_state[tox][toy] * current_player
                        if camp > 0:
                            break
                        elif camp < 0:
                            if tox == general_x:
                                return True
                            else:
                                break
            elif piece == 4:
                if general_x == x - 2 and current_state[x-1][y] == 0:
                    if general_y == y-1 or general_y == y+1:
                        return True
                if general_x == x + 2 and current_state[x+1][y] == 0:
                    if general_y == y - 1 or general_y == y + 1:
                        return True
                if general_y == y - 2 and current_state[x][y-1] == 0:
                    if general_x == x - 1 or general_x == x + 1:
                        return True
                if general_y == y + 2 and current_state[x][y+1] == 0:
                    if general_x == x - 1 or general_x == x + 1:
                        return True
            elif piece == 6:
                if x == general_x:
                    tox = x
                    interval = 0
                    for toy in range(y - 1, -1, -1):
                        camp = current_state[tox][toy] * current_player
                        if camp > 0:
                            interval += 1
                            if interval > 1:
                                break
                        elif camp < 0:
                            if interval == 1:
                                if toy == general_y:
                                    return True
                                else:
                                    break
                            elif interval > 1:
                                break
                            else:
                                interval += 1
                    interval = 0
                    for toy in range(y + 1, 9):
                        camp = current_state[tox][toy] * current_player
                        if camp > 0:
                            interval += 1
                            if interval > 1:
                                break
                        elif camp < 0:
                            if interval == 1:
                                if toy == general_y:
                                    return True
                                else:
                                    break
                            elif interval > 1:
                                break
                            else:
                                interval += 1
                elif y == general_y:
                    toy = y
                    interval = 0
                    for tox in range(x - 1, -1, -1):
                        camp = current_state[tox][toy] * current_player
                        if camp > 0:
                            interval += 1
                            if interval > 1:
                                break
                        elif camp < 0:
                            if interval == 1:
                                if tox == general_x:
                                    return True
                                else:
                                    break
                            elif interval > 1:
                                break
                            else:
                                interval += 1
                    interval = 0
                    for tox in range(x + 1, 10):
                        camp = current_state[tox][toy] * current_player
                        if camp > 0:
                            interval += 1
                            if interval > 1:
                                break
                        elif camp < 0:
                            if interval == 1:
                                if tox == general_x:
                                    return True
                                else:
                                    break
                            elif interval > 1:
                                break
                            else:
                                interval += 1
            elif piece == 7:
                if x == general_x - current_player and y == general_y:
                    return True
                elif x == general_x and (y == general_y-1 or y==general_y+1):
                    return True
    return False

def general_meet(current_state, self_general_position, opponent_general_position):
    is_meet=True
    if self_general_position[1] == opponent_general_position[1]:
        l1=int(self_general_position[0])
        l2=int(opponent_general_position[0])
        n=int(self_general_position[1])
        if l2-l1>0:
            step=1
        else:
            step=-1
        for l in range(l1+step,l2,step):
            if current_state[l][n]!= 0:
                is_meet=False
                break
    else:
        is_meet=False
    return is_meet