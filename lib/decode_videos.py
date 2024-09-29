import os
import tqdm

import base64


def decode_single_video(video_file_full_path, output_dir):
    with open(video_file_full_path, 'rb') as video:
        video_read = video.read()
    video_64_decode = base64.b64decode(video_read)
    ppg_video_path = os.path.join(output_dir, os.path.basename(video_file_full_path))

    # create a writable video and write the decoding result
    with open(ppg_video_path, 'wb') as video_result:
        video_result.write(video_64_decode)

    return ppg_video_path


def decode_all_video_files(input_dir, output_dir):
    for video_filename in tqdm.tqdm(os.listdir(input_dir)):

        if video_filename.endswith('.mp4'):
            video_file_full_path = os.path.join(input_dir, video_filename)
            decode_single_video(video_file_full_path, output_dir)


if __name__ == '__main__':
    input_dir = 'raw_videos'
    output_dir = 'videos'
    decode_all_video_files(input_dir, output_dir)
