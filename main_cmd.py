import argparse
from asciiAnime import AsciiProcess, AsciiVideoPlayer

def main():
    parser = argparse.ArgumentParser(description='Colorful Ascii Anime Player in your terminal!')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--width', type=int, default=50, help='Width of the ASCII art (default: 140)')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory to store cache files (default: ./cache)')
    parser.add_argument('--ratio', type=float, default=0.5, help='Aspect ratio of the ASCII art (default: 0.5)')
    
    args = parser.parse_args()

    ascii_process = AsciiProcess(args.video_path, args.width, args.cache_dir, args.ratio)
    ascii_path, color_path, metadata_path, audio_path = ascii_process.process()
    
    ascii_player = AsciiVideoPlayer(ascii_path, color_path, audio_path, metadata_path)
    ascii_player.play()

if __name__ == '__main__':
    main()