import cv2
import numpy as np
import os
import sys
import threading
import time
from tqdm import tqdm 
import miniaudio
from msvcrt import getwch
import ffmpeg

class AsciiProcess(object):
    def __init__(self, video_path, new_width, cache_dir=".", ratio=0.5, ASCII_CHARS = [".", ",", ":", ";", "+", "*", "?", "%", "S", "#", "@"]):
        self.video_path = video_path
        self.new_width = new_width
        self.ASCII_CHARS = ASCII_CHARS
        self.cache_dir = cache_dir
        self.ratio = ratio  # 字体宽高比
        
        self.video_name = os.path.basename(video_path).split(".")[0]
        
    def process(self):
        flag1 = False
        flag2 = False
        # 处理缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("缓存目录 {} 已存在，但不是一个目录".format(self.cache_dir))
            if not os.access(self.cache_dir, os.W_OK):
                raise ValueError("缓存目录 {} 不可写, 请重新设置缓存目录".format(self.cache_dir))
            if os.path.exists(os.path.join(self.cache_dir, f"{self.video_name}_ascii_{self.new_width}.npy")) and  \
            os.path.exists(os.path.join(self.cache_dir, f"{self.video_name}_color_{self.new_width}.npy")) and \
                os.path.exists(os.path.join(self.cache_dir, f"{self.video_name}_metadata_{self.new_width}.npy")):
                print(f"已存在处理好的字符画文件 {self.video_name}_ascii_{self.new_width}.npy, 跳过处理")
                print(f"已存在处理好的彩色帧文件 {self.video_name}_color.npy_{self.new_width}, 跳过处理")
                print(f"已存在处理好的元数据文件 {self.video_name}_metadata.npy_{self.new_width}, 跳过处理")
                flag1 = True
            if os.path.exists(os.path.join(self.cache_dir, f"{self.video_name}_audio_{self.new_width}.mp3")):
                print(f"已存在处理好的音频文件 {self.video_name}_audio_{self.new_width}.mp3, 跳过处理")
                flag2 = True
        
        # 处理音频和视频的时候，在终端显示旋转的光标
        def spinning_cursor():
            while True:
                for cursor in '|/-\\':
                    yield cursor

        spinner = spinning_cursor()

        # 定义一个标志来控制 spin 线程
        self.spin_flag = True

        def spin():
            while self.spin_flag:
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write('\b')

        # 处理音频部分
        if not flag2:
            print("开始处理音频部分 ...", end="")
            self.spin_flag = True
            spin_thread = threading.Thread(target=spin)
            spin_thread.start()
            
            audio_save_path = self.process_audio()
            self.spin_flag = False
            
            spin_thread.join()
        else:
            audio_save_path = os.path.join(self.cache_dir, f"{self.video_name}_audio.mp3")
            
        # 处理视频部分
        if not flag1:
            print("开始处理视频部分 ...", end="")
            ascii_save_path, color_save_path, metadata_save_path = self.process_video()
        else:
            ascii_save_path = os.path.join(self.cache_dir, f"{self.video_name}_ascii.npy")
            color_save_path = os.path.join(self.cache_dir, f"{self.video_name}_color.npy")
            metadata_save_path = os.path.join(self.cache_dir, f"{self.video_name}_metadata.npy")
        
        print("预处理完成")
        
        return ascii_save_path, color_save_path, metadata_save_path, audio_save_path
              
    def process_audio(self):
        """
        使用 ffmpeg 提取音频部分
        """
        audio_save_path = os.path.join(self.cache_dir, f"{self.video_name}_audio_{self.new_width}.mp3")
        # 直接使用 ffmpeg 提取音频流并保存为mp3文件
        # 这个需要保证 ffmpeg 已经安装在系统中,并且已经配置好环境变量
        ffmpeg.input(self.video_path).output(audio_save_path).run()
        print("音频已保存在 {}".format(audio_save_path))
        
        return audio_save_path
    
    def process_video(self):
        """
        处理视频部分
        """
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        ascii_frames = []
        color_frames = []
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
        ## 成功打开视频文件，开始逐帧处理
        for _ in tqdm(range(total_frames), desc="处理视频"):  # 使用tqdm显示进度条
            ret, frame = self.cap.read()
            if not ret:
                break
            
            resized_frame = self.resize_image(frame)
            grayscale_frame = self.grayify(resized_frame)
            ascii_str_array = self.pixels_to_ascii(grayscale_frame)
            
            ascii_frames.append(ascii_str_array)
            color_frames.append(resized_frame)
            
        ## 保存字符画处理结果
        ascii_frames = np.array(ascii_frames)
        ascii_save_path = os.path.join(self.cache_dir, f"{self.video_name}_ascii_{self.new_width}.npy")
        np.save(ascii_save_path, ascii_frames)
        
        ## 保存resize后的彩色帧 （存储颜色信息）
        color_frames = np.array(color_frames)
        color_save_path = os.path.join(self.cache_dir, f"{self.video_name}_color_{self.new_width}.npy")
        np.save(color_save_path, color_frames)
        print("彩色帧已保存在 {}".format(color_save_path))
        
        ## 保存视频元数据
        metadata = {
            "fps": fps,
            "frame_count": len(ascii_frames)
        }
        metadata_save_path = os.path.join(self.cache_dir, f"{self.video_name}_metadata_{self.new_width}.npy")
        np.save(metadata_save_path, metadata)
        
        self.cap.release()  # 释放视频文件
        print("字符画已保存在 {}".format(ascii_save_path))
        print("视频元数据已保存在 {}".format(metadata_save_path))   
        
        return ascii_save_path, color_save_path, metadata_save_path     
        
    def resize_image(self, image):
        """
        传入BGR图像，返回调整大小后的图像
        需要考虑图像的长宽比，字体宽高比
        rtype: ndarray(ndim=3, dtype=uint8)
        """
        height, width = image.shape[0], image.shape[1]  # cv 中的shape是(h, w, c)
        new_height = int(height / width * self.new_width * self.ratio)
        resized_image = cv2.resize(image, (self.new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_image
        
    def grayify(self, image):
        """
        传入BGR图像，返回灰度图像
        rtype: ndarray(ndim=2, dtype=uint8)
        """
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image

    def pixels_to_ascii(self, image):
        """
        传入灰度图像，返回ASCII字符画矩阵
        rtype: ndarray(ndim=2, dtype=str)
        """
        if image.max() - image.min() == 0:
            ascii_norm = np.zeros_like(image)
        else:
            ascii_norm = (image - image.min()) / (image.max() - image.min())
            
        ascii_indices = (ascii_norm * (len(self.ASCII_CHARS) - 1)).astype(int)
        ascii_str_array = np.array([self.ASCII_CHARS[idx] for idx in ascii_indices.flatten()]).reshape(ascii_indices.shape)
        return ascii_str_array
    

class AsciiVideoPlayer:
    def __init__(self, ascii_path, color_path, audio_path, metadata_path):
        self.ascii_path = ascii_path
        self.color_path = color_path
        self.audio_path = audio_path
        self.metadata_path = metadata_path
        
        self.ascii_frames = np.load(ascii_path)
        self.color_frames = np.load(color_path)
        
        metadata = np.load(metadata_path, allow_pickle=True).item()  # 由于保存的是python字典，需要设置allow_pickle=True
        self.frame_count = metadata["frame_count"]
        self.fps = metadata["fps"]
        
        # 同步控制参数
        self.frame_index = 0
        self.playing = False
        self.pause = False
        
        self.pause_cond = threading.Condition()
      
    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")  # 清屏命令,跨平台
        sys.stdout.flush()
        
    def hide_cursor(self):
        sys.stdout.write("\033[?25l")  # 隐藏光标
        
    def render_frame(self, frame_index):
        """
        在终端中渲染指定帧(彩色)，防止屏幕闪烁
        """
        # 使用 ANSI 转义序列移动光标而不是清屏 -> 避免屏幕闪烁
        sys.stdout.write("\033[H")  # 将光标移动到左上角
        ascii_frame = self.ascii_frames[frame_index]
        color_frame = self.color_frames[frame_index]
        
        for i, row in enumerate(ascii_frame):
            for j, char in enumerate(row):
                b, g, r = color_frame[i, j]
                sys.stdout.write(f"\033[38;2;{r};{g};{b}m{char}")
            sys.stdout.write("\n")
        sys.stdout.flush()

    def play_audio(self):
        stream = miniaudio.stream_file(self.audio_path)
        device = miniaudio.PlaybackDevice()
        device.start(stream)  # 非阻塞播放
        
        #! 播放音频的时候，需要用条件变量来控制设备的暂停和启动
        while self.playing:
            if self.pause:
                device.stop()
                with self.pause_cond:
                    self.pause_cond.wait()
                device.start(stream)    # 继续播放
            time.sleep(0.1)   # 减少 cpu 占用
        
        device.close()

    def play_video(self):
        self.playing = True
        self.clear_screen()  # 只在播放开始时清屏
        self.hide_cursor()  # 播放开始时隐藏光标
        
        start_time = time.perf_counter()
        
        #！ 播放视频时，此处可以使用简单的循环来控制播放和暂停，但是为了形式一致，使用了条件变量
        while self.playing:
            if self.pause:   # 暂停播放
                with self.pause_cond:
                    self.pause_cond.wait()
                start_time = time.perf_counter() - (self.frame_index / self.fps)  # 保持暂停时的时间
            
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            current_frame_index = int(elapsed_time * self.fps)
            if current_frame_index >= self.frame_count:  # 播放结束
                break
            if current_frame_index != self.frame_index:  # 渲染新帧,保持同步（可能打印的次数比实际帧数多，但是视觉上没有问题）
                self.render_frame(current_frame_index)
                self.frame_index = current_frame_index
        self.playing = False
    
    def play(self):
        # instructions
        print("按 p 暂停/播放")
        print("按 q 退出")
        input("按任意键开始播放")   # 阻塞等待开始播放
        
        # 两个线程同时开始
        audio_thread = threading.Thread(target=self.play_audio)
        audio_thread.start()
        video_thread = threading.Thread(target=self.play_video)
        video_thread.start()
        
        
        # 主线程等待用户信号
        while self.playing:   # 播放中,如果播放结束了，会自动退出循环
            try:
                user_input = getwch()  # 线程阻塞且无回显
                if user_input == "p":
                    self.pause = not self.pause
                    if not self.pause:
                        with self.pause_cond:
                            self.pause_cond.notify_all()
                elif user_input == "q":
                    self.playing = False
                    break
            except KeyboardInterrupt:
                self.playing = False
                break
        
        
        audio_thread.join()
        video_thread.join()
        
        print("播放结束")

if __name__ == "__main__":
    video_path = "./badapple.mp4"
    new_width = 100
    cache_dir = "./cache"
    ascii_process = AsciiProcess(video_path, new_width, cache_dir)
    ascii_path, color_path, metadata_path, audio_path = ascii_process.process()
    
    ascii_player = AsciiVideoPlayer(ascii_path, color_path, audio_path, metadata_path)
    
    ascii_player.play()
    print("播放结束, 按任意键退出")
    getwch()

