import copy
import os
import pickle
import time
import game_core as gc
from collections import deque
from collect_data import get_symmetry_data
import numpy as np
import requests
from bs4 import BeautifulSoup
from config import CONFIG
import multiprocessing
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from file_lock import FileLock

file_path = "fen_reptile.pkl"
lock_file = "fen_reptile.lock"

# 次数
repeat_times = 6


def create_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fen_to_state(fen):
    """Convert FEN string to state array"""
    state = np.zeros((10, 9), dtype=int)
    # Split by space and take only the board part (first part)
    board = fen.split(' ')[0]
    rows = board.split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                # Convert FEN piece to number
                piece_map = {
                    'k': 1, 'a': 2, 'b': 3, 'n': 4, 'r': 5, 'c': 6, 'p': 7,
                    'K': -1, 'A': -2, 'B': -3, 'N': -4, 'R': -5, 'C': -6, 'P': -7
                }
                state[i][col] = piece_map[char]
                col += 1
    return state


def get_state_array(state_deque, color):
    """Convert state deque to state array"""
    params = np.empty([17, 10, 9])
    for _i in range(16):
        params[_i] = copy.deepcopy(state_deque[_i - 16])
    params[16][:, :] = color
    return params


def fen_move_to_coord(fen_move, current_state):
    """Convert FEN move format (e.g. 'g8e2') to coordinate format (e.g. '0104')"""
    # Convert file (column) from a-h to 0-8
    file_map = {'a': '0', 'b': '1', 'c': '2', 'd': '3', 'e': '4',
                'f': '5', 'g': '6', 'h': '7', 'i': '8'}

    # In FEN format, the move is like 'g8e2' where:
    # First letter (g) is the from file
    # First number (8) is the from rank
    # Second letter (e) is the to file
    # Second number (2) is the to rank
    from_file = file_map[fen_move[0]]
    from_rank = str(9 - int(fen_move[1]))  # Invert the rank
    to_file = file_map[fen_move[2]]
    to_rank = str(9 - int(fen_move[3]))  # Invert the rank

    return from_rank + from_file + to_rank + to_file


class FenCollection:
    def __init__(self, url="https://www.xqipu.com", expand_url="/canjugupu/1554?page=", page_total=1, page=0, num=0,
                 min_step=0):
        self.url = url
        self.expand_url = expand_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3741.400 QQBrowser/10.5.3863.400'
        }
        self.page_total = page_total
        self.min_step = min_step
        self.buffer_size = CONFIG["buffer_size"]
        self.current_buffer_size = 0
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iterator = 0
        self.episode_len = 0
        self.session = create_session()

        self.page = page
        self.num = num

    def set_expand_url(self, expand_url, size):
        self.expand_url = expand_url
        self.page = 0
        self.num = -1
        self.page_total = size

    def do_search_one(self):
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                res = self.session.get(self.url + self.expand_url + str(self.page), headers=self.headers, timeout=30)
                res.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"页面不存在 (404): {self.url + self.expand_url + str(self.page)}")
                    return False
                if attempt == max_retries - 1:
                    print(f"获取页面失败,已重试{max_retries}次: {e}")
                    return False
                print(f"第{attempt + 1}次尝试失败: {e}. {retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            except (requests.exceptions.RequestException, OSError) as e:
                if attempt == max_retries - 1:
                    print(f"获取页面失败,已重试{max_retries}次: {e}")
                    return False
                print(f"第{attempt + 1}次尝试失败: {e}. {retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2

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
        if self.num < len(href_list) - 1:
            self.num = self.num + 1
        elif self.page < self.page_total - 1:
            self.page = self.page + 1
            self.num = 0
        else:
            return False

        for j in range(self.num, self.num + 1):
            print("============================== * 进度 * "
                  + str(self.page + 1) + "/" + str(self.page_total) + " * "
                  + str(j + 1) + "/" + str(len(href_list)) +
                  "* ================================")

            for attempt in range(max_retries):
                try:
                    res_children = self.session.get(self.url + href_list[j], headers=self.headers, timeout=30)
                    res_children.raise_for_status()
                    break
                except (requests.exceptions.RequestException, OSError) as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to fetch game data after {max_retries} attempts: {e}")
                        continue
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

            res_children.encoding = "utf-8"
            soup_children = BeautifulSoup(res_children.text, "html.parser")

            # Get initial FEN and moves from hidden elements
            init_fen = soup_children.find('div', {'id': 'qipu-init-fen'})
            moves = soup_children.find('div', {'id': 'qipu-moves-iccs'})

            if not init_fen or not moves:
                print("未找到对局数据！！！")
                continue

            init_fen = init_fen.text.strip()
            moves = moves.text.strip()

            print("初始盘面fen:", init_fen)

            # If no initial FEN is found, use the standard starting position
            if not init_fen:
                # init_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
                print("没有初始化盘面,跳过！！！")
                continue

            # Generate FEN list from initial FEN and moves
            fen_list = self.get_fen_list(init_fen, moves)

            if not fen_list:
                print("Failed to generate FEN list")
                continue

            # print(fen_list)

            title = soup_children.find('h1', {'class': 'page-header'})
            if title.text:
                print(title.text)
                print()

            result_div = soup_children.find('div', {'class': "qipu-basic"})
            result_children = result_div.find_all('div', {'class': "field--label-inline"})[6]
            result = result_children.find('div', {'class': 'field--item'}).text

            winner = -1

            print(result)
            if result[0:2] != "红先":
                continue
            elif result[2] == "胜":
                winner = -1
            elif result[2] == "负":
                winner = 1
            elif result[2] == "和":
                winner = 0
            else:
                continue

            play_data = []
            state_deque = deque(maxlen=16)

            # Initialize state deque with initial position
            initial_state = fen_to_state(fen_list[0])
            for _ in range(16):
                state_deque.append(initial_state)

            try:
                # Process each FEN position
                for i in range(1, len(fen_list)):
                    # print(state_deque[-1])
                    color = -1 if i % 2 == 1 else 1
                    # print(color)
                    state_array = get_state_array(state_deque, color)

                    # Create move probability array and set the corresponding move to 1
                    move_prob = np.zeros(2086, dtype=float)
                    if i < len(moves):
                        fen_move = moves[(i - 1) * 4:i * 4]  # Get the current move in FEN format
                        # Convert FEN move to coordinate format
                        coord_move = fen_move_to_coord(fen_move, state_deque[-1])
                        print(coord_move)
                        move_id = gc.action_to_id_mapping.get(coord_move)
                        if move_id is not None:
                            move_prob[move_id] = 1.0
                        else:
                            raise Exception("落子动作解析失败！")
                    # Add the position to play_data
                    play_data.append((state_array, move_prob, winner * color))

                    current_state = fen_to_state(fen_list[i])
                    state_deque.append(current_state)
            except:
                continue
            if len(play_data) >= self.min_step:
                print()
                print("最终:")
                print(state_deque[-1])
                self.collect_data(play_data)
                return True
            else:
                print("舍弃 - 步数不足")
                return False
        return False

    def get_fen_list(self, init_fen, moves):
        """Generate FEN list from initial FEN and moves"""
        fen_list = [init_fen]
        current_fen = init_fen

        # Process moves in pairs (each move is 4 characters)
        for i in range(0, len(moves), 4):
            if i + 4 > len(moves):
                break
            move = moves[i:i + 4]
            next_fen = self.next_fen(current_fen, move)
            if next_fen:
                fen_list.append(next_fen)
                current_fen = next_fen
            else:
                print(f"Failed to process move: {move}")
                return None

        return fen_list

    def next_fen(self, fen, move):
        """Calculate next FEN position based on current FEN and move"""
        if not fen or not move or len(move) != 4:
            return None

        # Split FEN into board and turn
        board, turn = fen.split()

        # Convert board to array
        board_array = []
        for row in board.split('/'):
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend([''] * int(char))
                else:
                    board_row.append(char)
            board_array.append(board_row)

        # Convert ICCS coordinates (a-i, 0-9) to array indices
        x_map = {
            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
            'f': 5, 'g': 6, 'h': 7, 'i': 8
        }

        try:
            from_x = x_map[move[0]]
            from_y = 9 - int(move[1])  # Convert from top-down to bottom-up
            to_x = x_map[move[2]]
            to_y = 9 - int(move[3])  # Convert from top-down to bottom-up

            # Make move
            piece = board_array[from_y][from_x]
            if not piece:
                print(f"No piece at source position: {move}")
                return None

            board_array[from_y][from_x] = ''
            board_array[to_y][to_x] = piece

            # Convert array back to FEN
            new_board = []
            for row in board_array:
                fen_row = ''
                empty_count = 0
                for cell in row:
                    if cell == '':
                        empty_count += 1
                    else:
                        if empty_count > 0:
                            fen_row += str(empty_count)
                            empty_count = 0
                        fen_row += cell
                if empty_count > 0:
                    fen_row += str(empty_count)
                new_board.append(fen_row)

            # Switch turn
            new_turn = 'b' if turn == 'w' else 'w'

            return '/'.join(new_board) + ' ' + new_turn

        except (KeyError, ValueError) as e:
            print(f"Invalid move coordinates: {move}, error: {e}")
            return None

    def collect_data(self, play_data):
        self.episode_len = len(play_data)
        to_save_play_data = get_symmetry_data(play_data)
        
        # 使用文件锁保护文件访问
        with FileLock(CONFIG["train_data_path"]) as lock:
            if os.path.exists(CONFIG["train_data_path"]):
                try:
                    with open(CONFIG["train_data_path"], "rb") as data_dict:
                        data_file = pickle.load(data_dict)
                        self.data_buffer = data_file["data_buffer"]
                        self.iterator = data_file["iterator"]
                        del data_file
                        self.iterator += 1
                        self.data_buffer.extend(to_save_play_data)
                    print('成功载入数据')
                except Exception as e:
                    print(f'载入数据失败: {e}')
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

    def run(self, href):
        try:
            flag = False
            while flag == False:
                flag = self.do_search_one()
                data_dict = {"href": href, "page": self.page, "num": self.num}
                with open(file_path, "wb") as data_file:
                    pickle.dump(data_dict, data_file)
                if flag == False and self.page >= self.page_total - 1:
                    return False
            return flag
        except KeyboardInterrupt:
            print('\n\rquit')


def get_list():
    # 创建锁文件
    if os.path.exists(lock_file):
        print("另一个进程正在运行，等待...")
        while os.path.exists(lock_file):
            time.sleep(5)

    # 创建锁文件
    with open(lock_file, 'w') as f:
        f.write(str(os.getpid()))

    try:
        session = create_session()
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3741.400 QQBrowser/10.5.3863.400'
        }
        href_list = ["/canjugupu/1555", "/canjugupu/1561", "/canjugupu/1547", "/canjugupu/1554", "/canjugupu/1549",
                     "/canjugupu/27571", "/canjugupu/27572", "/canjugupu/27573", "/canjugupu/4330", "/canjugupu/1553",
                     "/canjugupu/1572", "/canjugupu/1550", "/canjugupu/1620", "/canjugupu/1610", "/canjugupu/1615"]
        data_dict = {}
        index = 0
        if os.path.exists(file_path):
            while True:
                try:
                    with open(file_path, "rb") as data_dict:
                        data_dict = pickle.load(data_dict)
                    break
                except:
                    time.sleep(5)
        else:
            data_dict = {"href": href_list[0], "page": 0, "num": -1}
        href = None
        for j in range(0, len(href_list)):
            if data_dict["href"] == href_list[j]:
                index = j
                href = href_list[j]
        if href == None:
            href = href_list[0]
            data_dict["page"] = 0
            data_dict["num"] = -1

        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                page_size = session.get("https://www.xqipu.com" + href, headers=header, timeout=30)
                page_size.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"页面不存在 (404): https://www.xqipu.com{href}")
                    # 移动到下一个URL
                    if index + 1 < len(href_list):
                        index += 1
                        href = href_list[index]
                        data_dict["page"] = 0
                        data_dict["num"] = -1
                        print(f"尝试下一个URL: {href}")
                        continue
                    else:
                        print("没有更多URL可尝试")
                        return
                if attempt == max_retries - 1:
                    print(f"获取页面失败,已重试{max_retries}次: {e}")
                    return
                print(f"第{attempt + 1}次尝试失败: {e}. {retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            except (requests.exceptions.RequestException, OSError) as e:
                if attempt == max_retries - 1:
                    print(f"获取页面失败,已重试{max_retries}次: {e}")
                    return
                print(f"第{attempt + 1}次尝试失败: {e}. {retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2

        page_size_res = BeautifulSoup(page_size.text, "html.parser")
        pagination = page_size_res.find('ul', {'class': 'pagination'})
        if pagination:
            li_last = pagination.find('li', 'pager-last')
            if li_last:
                short_href = li_last.find('a').get('href')
                str_list = short_href.split('=')
                size = int(str_list[1]) + 1
            else:
                size = 1
        else:
            size = 1
        collect = FenCollection(url="https://www.xqipu.com", expand_url=href + "?page=", page_total=size,
                                page=data_dict["page"], num=data_dict["num"], min_step=1)
        for k in range(0, repeat_times):
            once = collect.run(href)
            if once == False and index + 1 < len(href_list):
                index = index + 1
                href = href_list[index]
                print(f"切换到下一个URL: {href}")
                for attempt in range(max_retries):
                    try:
                        page_size = session.get("https://www.xqipu.com" + href, headers=header, timeout=30)
                        page_size.raise_for_status()
                        page_size_res = BeautifulSoup(page_size.text, "html.parser")
                        pagination = page_size_res.find('ul', {'class': 'pagination'})
                        if pagination:
                            li_last = pagination.find('li', 'pager-last')
                            if li_last:
                                short_href = li_last.find('a').get('href')
                                str_list = short_href.split('=')
                                size = int(str_list[1]) + 1
                            else:
                                size = 1
                        else:
                            size = 1
                        collect.set_expand_url(href + "?page=", size)
                        once = collect.run(href)
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            print(f"页面不存在 (404): https://www.xqipu.com{href}")
                            # 移动到下一个URL
                            if index + 1 < len(href_list):
                                index += 1
                                href = href_list[index]
                                data_dict["page"] = 0
                                data_dict["num"] = -1
                                continue
                            else:
                                print("没有更多URL可尝试")
                                return
                        if attempt == max_retries - 1:
                            print(f"获取页面失败,已重试{max_retries}次: {e}")
                            return
                        print(f"第{attempt + 1}次尝试失败: {e}. {retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    except (requests.exceptions.RequestException, OSError) as e:
                        if attempt == max_retries - 1:
                            print(f"获取页面失败,已重试{max_retries}次: {e}")
                            return
                        print(f"第{attempt + 1}次尝试失败: {e}. {retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
    finally:
        # 确保在函数结束时删除锁文件
        if os.path.exists(lock_file):
            os.remove(lock_file)


if __name__ == "__main__":
    for l in range(1, 2):
        get_list() 