import copy
import os
import pickle
import time
from collections import deque
import game_core as gc
import numpy as np
import requests
from bs4 import BeautifulSoup
from config import CONFIG


chess_string_to_num={'帅':1,'将':1,'仕':2,'士':2,'相':3,'象':3,'马':4,'车':5,'炮':6,'兵':7,'卒':7}
number_mapping={'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'一':8,'二':7,'三':6,'四':5,'五':4,'六':3,'七':2,'八':1,'九':0}
move_num_number_mapping={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
move_long={'马':3,'相':4,'象':4,'仕':2,'士':2}
def find_chess(name,point_y,state,color):
    chess=None
    for x in range(10):
        if state[x][point_y]==chess_string_to_num[name]*color:
            if chess is not None:
                return None
            chess=str(x)+str(point_y)
    return chess
def get_action(name,position,move,color):
    tox = int(position[0])
    toy = int(position[1])
    if name not in chess_string_to_num:
        return None
    elif name=="马" or name=="相" or name=="象" or name=="仕" or name=="士":
        difference=abs(toy-number_mapping[move[1]])
        if move[0]=="进":
            tox+=color*(move_long[name]-difference)
            toy=number_mapping[move[1]]
        elif move[0]=="退":
            tox -= color*(move_long[name]-difference)
            toy = number_mapping[move[1]]
        else:
            return None
    else:
        if move[0] == "进":
            tox += color * move_num_number_mapping[move[1]]
        elif move[0] == "退":
            tox -= color * move_num_number_mapping[move[1]]
        elif move[0] == "平":
            toy = number_mapping[move[1]]
        else:
            return None
    if tox<0 or tox>9 or toy<0 or toy>8:
        return None
    return str(tox) + str(toy)

def get_state_array(state_deque,color):
    params = np.empty([17, 10, 9])
    for _i in range(16):
        params[_i] = copy.deepcopy(state_deque[_i - 16])
    params[16][:, :] = color
    return params


#"/qipus?page="
class Collection:
    def __init__(self,url="https://www.xqipu.com",expand_url="/canjugupu/1554?page=",page_total=1,min_step=60):
        self.url=url
        self.expand_url=expand_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3741.400 QQBrowser/10.5.3863.400'
        }
        self.page_total = page_total
        self.min_step=min_step
        self.buffer_size = CONFIG["buffer_size"]  # 经验池大小
        self.current_buffer_size = 0
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iterator = 0
        self.episode_len = 0

    def do_search(self,page):
        res = requests.get(self.url + self.expand_url + str(page),headers=self.headers)
        soup = BeautifulSoup(res.text, "html.parser")
        hrefs = soup.find_all("span", {'class': 'field-content'})
        href_list = []
        for i in range(0, len(hrefs)):
            try:
                a = hrefs[i].find('a')
                if a is not None:
                    href_list.append(a.get('href'))
            except:
                pass
        for j in range(0, len(href_list)):
            print("============================== * 进度 * "
                  + str(page + 1) + "/" + str(self.page_total) + " * "
                  + str(j + 1) + "/" + str(len(href_list)) +
                  "* ================================")
            res_children = requests.get(self.url + href_list[j],headers=self.headers)
            res_children.encoding = "utf-8"
            soup_children = BeautifulSoup(res_children.text, "html.parser")
            title=soup_children.find('h1',{'class':'page-header'})
            if title.text:
                print(title.text)
                print()

            result_div = soup_children.find('div', {'class': "qipu-basic"})
            result_children = result_div.find_all('div', {'class': "field--label-inline"})[6]
            result = result_children.find('div', {'class': 'field--item'}).text
            if result[0:2] != "红先":
                continue
            elif result[2] == "胜":
                winner = -1
            elif result[2] == "负":
                winner = 1
            elif result[2] == "和":
                continue
            else:
                continue
            state_deque = deque(maxlen=16)
            play_data = []
            for _ in range(16):
                state_deque.append(np.array(gc.init_state))
            uls = soup_children.find("ul", {'id': 'moves_text'})
            lis = uls.find_all('li', {'class': 'round'})
            for k in range(0, len(lis)):
                span1 = lis[k].find('span', {'class': 'move', 'name': str((k + 1) * 2 - 1)})
                span2 = lis[k].find('span', {'class': 'move', 'name': str((k + 1) * 2)})
                red, black = None, None
                if span1:
                    red = span1.text
                if span2:
                    black = span2.text
                if not red or red[1] not in number_mapping:
                    play_data = None
                    break
                red_chess = find_chess(red[0], number_mapping[red[1]], state_deque[-1], -1)
                if red_chess is None:
                    play_data = None
                    print("重复")
                    break
                red_action = get_action(red[0], red_chess, red[2:4], -1)
                if red_action is None:
                    play_data = None
                    break
                red_state = get_state_array(state_deque, -1)
                red_move = red_chess + red_action
                state_deque.append(gc.state_change_by_move(state_deque[-1], red_move))

                red_move_prob = np.zeros(2086, dtype=float)
                red_move_prob[gc.action_to_id_mapping[red_move]] = 1.0
                play_data.append((red_state,red_move_prob,winner*-1.0))
                print(red)
                print(state_deque[-1])

                if not black or black[1] not in number_mapping:
                    play_data=None
                    break
                black_chess = find_chess(black[0], number_mapping[black[1]], state_deque[-1], 1)
                if black_chess is None:
                    play_data = None
                    print("重复")
                    break
                black_action = get_action(black[0], black_chess, black[2:4], 1)
                if black_action is None:
                    play_data = None
                    break
                black_state = get_state_array(state_deque, 1)
                black_move = black_chess + black_action
                state_deque.append(gc.state_change_by_move(state_deque[-1], black_move))

                black_move_prob = np.zeros(2086, dtype=float)
                black_move_prob[gc.action_to_id_mapping[black_move]] = 1.0
                play_data.append((black_state, black_move_prob, winner * 1.0))

                print(black)
                print(state_deque[-1])

            if play_data is not None and len(play_data)>=self.min_step:
                print()
                print("最终:")
                print(state_deque[-1])
            elif play_data is not None:
                play_data=None
            else:
                print("舍弃")

            if play_data:
                self.collect_data(play_data)

    def collect_data(self, play_data):
        self.episode_len = len(play_data)
        if os.path.exists(CONFIG["train_data_path"]):
            while True:
                try:
                    with open(CONFIG["train_data_path"], "rb") as data_dict:
                        data_file = pickle.load(data_dict)
                        self.data_buffer = data_file["data_buffer"]
                        self.iterator = data_file["iterator"]
                        del data_file
                        self.iterator += 1
                        self.data_buffer.extend(play_data)
                    print('成功载入数据')
                    break
                except:
                    time.sleep(5)
        else:
            self.data_buffer.extend(play_data)
            self.iterator += 1
        self.current_buffer_size = len(self.data_buffer)
        data_dict = {"data_buffer": self.data_buffer, "iterator": self.iterator}
        with open(CONFIG["train_data_path"], "wb") as data_file:
            pickle.dump(data_dict, data_file)
        print('batch i: {},buffer current/max: {}/{}, episode_len: {}'.format(
            self.iterator, self.current_buffer_size, self.buffer_size, self.episode_len))
        time.sleep(2)
    def run(self):
        try:
            for num in range(0, self.page_total):
                self.do_search(num)
        except KeyboardInterrupt:
            print('\n\rquit')

def get_list(page):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3741.400 QQBrowser/10.5.3863.400'
    }
    page_list = requests.get("https://www.xqipu.com/eventlist?page="+str(page),headers= header)
    page_res = BeautifulSoup(page_list.text, "html.parser")
    page_res.encoding = "utf-8"
    table = page_res.find('table',{'class':'views-table'})
    tr_list = table.find_all('td',{'class':'views-field'})
    href_list = []
    for i in range(len(tr_list)):
        try:
            a = tr_list[i].find('a')
            if a is not None:
                href_list.append(a.get('href'))
        except:
            pass
    for j in range(0, len(href_list)):
        page_size = requests.get("https://www.xqipu.com"+href_list[j], headers= header)
        page_size_res = BeautifulSoup(page_size.text, "html.parser")
        pagination=page_size_res.find('ul',{'class':'pagination'})
        if pagination:
            li_last=pagination.find('li','pager-last')
            if li_last :
                short_href=li_last.find('a').get('href')
                str_list=short_href.split('=')
                size=int(str_list[1])+1
            else:
                size=1
        else:
            size=1
        collect=Collection(url="https://www.xqipu.com",expand_url=href_list[j]+"?page=",page_total=size,min_step=50)
        collect.run()

if __name__=="__main__":
    # collect=Collection(url="https://www.xqipu.com",expand_url="/eventqipu/27518?page=",page_total=9,min_step=50)
    # collect.run()

    for l in range(1,5):
        get_list(l)

