#!/usr/bin/env python3

# 필요한 모듈 import
import os
import yaml
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import rclpy
import cv2
import numpy as np
import time

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, UInt8

# Node 클래스를 상속하여 LineFollower 클래스 정의
class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        package_name = 'line_follower'

        # 이미지 처리를 위한 변수 및 객체 생성
        self.bridge = CvBridge()
        self.twist = Twist()

        # subscription, publisher 생성
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.process_image, 10)
        self.image_pub = self.create_publisher(Image, '/processed_image', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.pub_image_lane = self.create_publisher(CompressedImage, '/detect/line_image/compressed', 1)

        self.pub_white_line_reliability = self.create_publisher(UInt8, '/detect/white_line_reliability', 1)
        self.pub_yellow_line_reliability = self.create_publisher(UInt8, '/detect/yellow_line_reliability', 1)
        
        self.pub_image_white_lane = self.create_publisher(Image, '/detect/white_line_image', 1)
        self.pub_image_yellow_lane = self.create_publisher(Image, '/detect/yellow_line_image', 1)


        # 주행 제어를 위한 변수를 초기화합니다
        self.lastError = 0.0
        self.MAX_VEL = 0.12
        self.counter = 1
        self.reliability_white_line = 100
        self.reliability_yellow_line = 100
        self.mov_avg_left = np.zeros((0, 3))
        self.mov_avg_right = np.zeros((0, 3))
        self.right_fitx = None
        self.left_fitx = None
        self.lane_fit_bef = None

        # 카메라 캘리브레이션 파일로부터 카메라 내부 파라미터와 왜곡 계수를 불러옵니다.
        params_file = os.path.join(get_package_share_directory(package_name), 'param', 'camera_params.yaml')
        self.K, self.D = self.load_camera_parameters(params_file)

    def load_camera_parameters(self,params_file):
        """
        주어진 YAML 파일에서 카메라 매개 변수를 로드하는 함수.

        Args:
            params_file: 로드할 YAML 파일 경로(str).

        Returns:
            K: 카메라 캘리브레이션 행렬(numpy.ndarray).
            D: 카메라 왜곡 계수(numpy.ndarray).
        """
        # 파일 열기
        with open(params_file, 'r') as f:
            # yaml 형식으로 로드
            params = yaml.load(f, Loader=yaml.FullLoader)

        # 필요한 카메라 매트릭스와 왜곡 계수 추출
        K = params['narrow_stereo']['camera_matrix']['data']
        D = params['narrow_stereo']['distortion_coefficients']['data']

        # 배열로 변환 후 크기 조정
        K = np.array(K).reshape((3, 3))
        D = np.array(D)
        
        return K, D

    def process_image(self, msg):
        """
        주어진 CompressedImage 메시지를 받아서 처리하는 함수.

        Args:
            msg: CompressedImage 메시지.

        Returns:
            None
        """
        # 카운터가 3의 배수가 아니면 건너뛰기
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        # CompressedImage 메시지를 디코딩하여 이미지를 복원.
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        #이미지 캡처 후 저장
        cv2.imwrite('processed_image.jpg', image) 

        #카메라 켈리 파일 적용
        #image = cv2.undistort(image, self.K, self.D)

        # 이미지의 크기 지정
        height, width = image.shape[:2]

        # 좌상단, 우상단, 우하단, 좌하단 좌표를 정의. 
        src_points = np.float32([[45,197 ], [209,194 ], [252,238 ], [6,237]])

        # 변환할 좌표를 정의.
        dst_points = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])

        # 변환 행렬을 계산.
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 이미지를 원근 변환.
        warped = cv2.warpPerspective(image, M, (width, height))

        #변수 초기화
        yellow_fraction = 0
        white_fraction = 0

        # 횡단보도를 검출 안한 경우 하얀선과 노란선을 검출 실행.
        if not self.detect_crosswalk(warped):
            white_fraction, cv_white_lane = self.maskWhiteLane(warped)
            yellow_fraction, cv_yellow_lane = self.maskYellowLane(warped)
        else:
            # 검출을 한 경우 crossDown 실행
            self.cmd_vel(0.075, -0.025)
            time.sleep(2)
            self.cmd_vel(0.0, 0.0)
            time.sleep(3)
        try:
            #노란색 차선에 대한 처리
            if yellow_fraction > 150:
                # 현재 검출된 노란색 차선으로부터 좌표값을 추출.
                self.left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, cv_yellow_lane)
                # 이동 평균 값을 계산하기 위해 이전 좌표값에 현재 좌표값을 저장.
                self.mov_avg_left = np.append(self.mov_avg_left,np.array([self.left_fit]), axis=0)
            # 흰색 차선에 대한 처리
            if white_fraction > 150:
                # 현재 검출된 하얀색 차선으로부터 좌표값을 추출.
                self.right_fitx, self.right_fit = self.fit_from_lines(self.right_fit, cv_white_lane)
                # 이동 평균 값을 계산하기 위해 이전 좌표값에 현재 좌표값을 저장.
                self.mov_avg_right = np.append(self.mov_avg_right,np.array([self.right_fit]), axis=0)
        except:
            # 예외 처리 - 노란색 차선
            if yellow_fraction > 150:
                # 검출된 노란색 차선이 없는 경우 슬라이딩 윈도우 방법으로 차선을 추출.
                self.left_fitx, self.left_fit = self.sliding_windown(cv_yellow_lane, 'left')
                # 이동 평균 값을 계산하기 위해 이전 좌표값에 현재 좌표값을 저장.
                self.mov_avg_left = np.array([self.left_fit])
            # 예외 처리 - 흰색 차선
            if white_fraction > 150:
                # 검출된 흰색 차선이 없는 경우 슬라이딩 윈도우 방법으로 차선을 추출.
                self.right_fitx, self.right_fit = self.sliding_windown(cv_white_lane, 'right')
                # 이동 평균 값을 계산하기 위해 현재 좌표값을 저장.
                self.mov_avg_right = np.array([self.right_fit])

        #이동 평균 길이 설정
        MOV_AVG_LENGTH = 5

        #왼쪽 차선에 대한 이동 평균 계산
        self.left_fit = np.array([np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
                            np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
                            np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])])
        #오른쪽 차선에 대한 이동 평균 계산
        self.right_fit = np.array([np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
                            np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
                            np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])])

        #이동 평균 배열 길이가 100보다 클 영우 이동 평균 길이로 자르기
        if self.mov_avg_left.shape[0] > 100:
            self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]
        if self.mov_avg_right.shape[0] > 100:
            self.mov_avg_right = self.mov_avg_right[0:MOV_AVG_LENGTH]

        #make_lane 함수 호출
        self.make_lane(warped, white_fraction, yellow_fraction)  

    def detect_crosswalk(self, image):
        """
        주어진 이미지에서 횡단보도를 검출하는 함수.

        Args:
            image: 검출할 이미지(numpy.ndarray).

        Returns:
            bool: 검출된 횡단보도가 6개 이상이면 True를, 그렇지 않으면 False를 반환.
        """
        # 횡단보도 개수를 저장할 변수
        crosswalk_count = 0 

        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 그레이스케일 이미지에 임계값(threshold)을 적용하여 이진화
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # 윤곽선(contour)을 찾기 위해 findContours 함수를 사용
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 윤곽선(contour)을 하나씩 순회하며 횡단보도 형태를 확인
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 이미지에서 횡단보도 크기에 맞게 조절
            if 500 < area <1000 : 
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # 종횡비(aspect ratio)가 일정 범위 내에 있는지 확인
                if 0.06 < aspect_ratio < 0.15:
                     # 검출된 횡단보도 주변에 사각형을 그리기, 카운터 1 증가
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    crosswalk_count += 1

        # 검출된 횡단보도 개수가 6개 이상이면 True 반환, 그렇지 않으면 False 반환
        if crosswalk_count >= 6:
            return True

        
        return False  

    def maskWhiteLane(self, image):
        """
        주어진 이미지에서 하얀색 차선을 검출하기 위해 색상 필터링을 수행하는 함수.

        Args:
            image: 검출할 이미지(numpy.ndarray).

        Returns:
            fraction_num: 검출된 하얀색 차선의 수직 방향 비율을 의미하는 변수.
            mask: 검출된 하얀색 차선에 해당하는 마스크 이미지.
        """
        # BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 하얀색 차선의 HSV 범위를 정의
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 44, 255])

        # 하얀색에 해당하는 픽셀만을 가진 마스크 이미지를 생성.
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 원본 이미지에서 하얀색에 해당하는 픽셀만 추출.
        res = cv2.bitwise_and(image, image, mask = mask)

        # mask 값에서 0이 아닌 값의 개수 계산 후 fraction_num 저장
        fraction_num = np.count_nonzero(mask)

        # 초기화
        how_much_short = 0

        # mask 에서 240행 중 0이 아닌 값이 있는 행 수 how_much_short 저장
        for i in range(0, 240):
            if np.count_nonzero(mask[i,::]) > 0:
                how_much_short += 1

        # 240개 행 중에 0이 아닌 값이 없는 행 수 how_much_short 저장
        how_much_short = 240 - how_much_short

        if how_much_short > 100:
            # 검출된 차선이 일부분 끊겼거나 검출이 잘 되지 않은 경우
            if self.reliability_white_line >= 5:
                self.reliability_white_line -= 5
        elif how_much_short <= 100:
            # 검출된 차선이 잘 된 경우 
            if self.reliability_white_line <= 99:
                self.reliability_white_line += 5

        # 검출된 차선의 신뢰도를 ROS 메시지로 publish
        msg_white_line_reliability = UInt8()
        msg_white_line_reliability.data = self.reliability_white_line
        self.pub_white_line_reliability.publish(msg_white_line_reliability)

        # 검출된 차선 이미지 발행 
        msg = self.mask_lane(mask)
        self.pub_image_white_lane.publish(msg)
        
        return fraction_num, mask

    def maskYellowLane(self, image):
        """
        주어진 이미지에서 노란색 차선을 검출하기 위해 색상 필터링을 수행하는 함수.

        Args:
            image: 검출할 이미지(numpy.ndarray).

        Returns:
            fraction_num: 검출된 노란색 차선의 수직 방향 비율을 의미하는 변수.
            mask: 검출된 노란색 차선에 해당하는 마스크 이미지.
        """
        # BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 노란색 차선의 HSV 범위를 정의
        lower_yellow = np.array([20, 50, 95])
        upper_yellow = np.array([45, 255, 255])

        # 노란색에 해당하는 픽셀만을 가진 마스크 이미지를 생성.
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 원본 이미지에서 노란색에 해당하는 픽셀만 추출.
        res = cv2.bitwise_and(image, image, mask = mask)
        
        # mask 값에서 0이 아닌 값의 개수 계산 후 fraction_num 저장
        fraction_num = np.count_nonzero(mask)

        #초기화
        how_much_short = 0

        #mask 에서 240행 중 0이 아닌 값이 있는 행 수  how_much_short 저장
        for i in range(0, 240):
            if np.count_nonzero(mask[i,::]) > 0:
                how_much_short += 1
        # 240개 행 중에 0이 아닌 값이 없는 행 수 how_much_short 저장
        how_much_short = 240 - how_much_short

        if how_much_short > 100:
            #검출된 차선이 일부분 끊겼거나 검출이 잘 되지 않은 경우
            if self.reliability_yellow_line >= 5:
                self.reliability_yellow_line -= 5
        elif how_much_short <= 100:
            #검출된 차선이 잘 된 경우 
            if self.reliability_yellow_line <= 99:
                self.reliability_yellow_line += 5
                
        # 검출된 차선의 신뢰도를 ROS 메시지로 publish
        msg_yellow_line_reliability = UInt8()
        msg_yellow_line_reliability.data = self.reliability_yellow_line
        self.pub_yellow_line_reliability.publish(msg_yellow_line_reliability)

        # 검출된 차선을 이미지 형식으로 publish
        msg = self.mask_lane(mask)
        self.pub_image_yellow_lane.publish(msg)

        return fraction_num, mask

    def mask_lane(self, img):
        """
        검출된 차선 이미지 메시지 생성하는 함수

        Parameters
        ----------
        img : numpy.ndarray
            차선 검출 결과 이미지

        Returns
        -------
        msg : sensor_msgs.msg.Image
            차선 검출 결과 이미지 메시지
        """
        # 검출된 차선 이미지 메시지 생성
        msg = Image() 
        # 메시지 발행 시간 설정
        msg.header.stamp = self.get_clock().now().to_msg() 
        # 이미지 세로 길이 설정
        msg.height = img.shape[0]
        # 이미지 가로 길이 설정 
        msg.width = img.shape[1]
        # 이미지 색상 채널 설정 
        msg.encoding = "rgb8"
        # 엔디안 설정
        msg.is_bigendian = False 
        # 이미지 한 줄당 바이트 수 설정
        msg.step = 3 * img.shape[1] 
        # 이미지 데이터 설정
        msg.data = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).tobytes() 

        return msg

    def fit_from_lines(self, lane_fit, image):
        """
        현재 검출된 라인 주변의 흰색, 노란색 픽셀 영역을 이용하여 라인을 추정하는 함수.

        Args:
            lane_fit: 현재 검출된 라인에 대한 정보가 저장된 변수.
            image: 검출 대상 이미지(numpy.ndarray).

        Returns:
            lane_fitx: 추정된 라인에 대한 정보가 저장된 변수.
            lane_fit: 추정된 라인에 대한 정보가 저장된 변수.
        """
        # nonzero() 함수를 이용하여 마스크 이미지에서 흰색 영역의 픽셀 좌표를 추출.
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 라인을 검출하기 위한 마진을 설정.
        margin = 100

        # lane_inds는 마스크 이미지에서 흰색,노란 영역 중에서 현재 검출된 라인 주변에 있는 픽셀들의 인덱스를 저장.
        lane_inds = ((nonzerox > (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) & (
        nonzerox < (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin)))

        # 인덱스를 이용하여 검출된 라인 주변에 있는 픽셀의 좌표를 추출.
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # 추출된 좌표를 이용하여 2차 함수를 이용하여 새로운 라인을 추정.
        lane_fit = np.polyfit(y, x, 2)

        # 추정된 라인으로부터 픽셀 좌표에 해당하는 y좌표 값을 이용하여 x좌표 값을 계산.
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

          
        return lane_fitx, lane_fit

    def sliding_windown(self, img_w, left_or_right):
        """
        슬라이딩 윈도우 방식을 이용하여 차선을 검출하는 함수.

        Args:
            img_w: 슬라이딩 윈도우를 적용할 이미지(numpy.ndarray).
            left_or_right: 추정하고자 하는 차선이 왼쪽인지 오른쪽인지를 나타내는 문자열('left' or 'right').

        Returns:
            lane_fitx: 추정된 차선의 x 좌표에 해당하는 값(numpy.ndarray).
            lane_fit: 추정된 차선의 2차 함수 계수(numpy.ndarray).
        """
        # 이미지의 아래 절반부터 흰색 픽셀 값을 기준으로 히스토그램을 생성.
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)

        # 시각화를 위한 빈 이미지를 생성.
        out_img = np.dstack((img_w, img_w, img_w)) * 255

       # 히스토그램에서 왼쪽 또는 오른쪽 영역에서 가장 높은 값을 찾아 시작점을 설정.
        midpoint = np.int(histogram.shape[0] / 2)

        if left_or_right == 'left':
            lane_base = np.argmax(histogram[:midpoint])
        elif left_or_right == 'right':
            lane_base = np.argmax(histogram[midpoint:]) + midpoint

        # 슬라이딩 윈도우의 수를 지정.
        nwindows = 10

        # 슬라이딩 윈도우의 높이를 지정.
        window_height = np.int(img_w.shape[0] / nwindows)

        # 이미지에서 흰색 픽셀의 x, y 좌표를 검출.
        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 슬라이딩 윈도우마다 현재 위치를 업데이트.
        x_current = lane_base

        # 슬라이딩 윈도우의 너비를 지정.
        margin = 50

        # 유효한 픽셀 수가 일정 수 이상일 때 윈도우의 중심을 조정.
        minpix = 50

        # 차선을 구성하는 좌표를 저장할 리스트를 생성.
        lane_inds = []

        # 슬라이딩 윈도우를 하나씩 탐색.
        for window in range(nwindows):
            # 슬라이딩 윈도우의 경계를 x와 y 좌표로 식별.
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # 시각화 이미지에 슬라이딩 윈도우 그림.
            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            # x와 y에서 0이 아닌 픽셀 위치를 식별.
            good_lane_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                nonzerox < win_x_high)).nonzero()[0]

            # 이 인덱스를 리스트에 추가.
            lane_inds.append(good_lane_inds)

            # minpix 이상의 픽셀을 찾았다면 평균 위치를 사용하여 다음 창의 위치를 조정.
            if len(good_lane_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_lane_inds]))

        # 인덱스 배열을 연결.
        lane_inds = np.concatenate(lane_inds)

        # 선 픽셀 위치를 추출.
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        try:
             # 2차 방정식으로 라인 픽셀 위치를 설정.
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except:
            if self.lane_fit_bef is None:
                # 이전에 저장된 값이 없으면 [0, 0, 2]를 사용.
                lane_fit = np.array([0, 0, 2])
            else:
                # 이전에 저장된 값이 있으면 그 값을 사용.
                lane_fit = self.lane_fit_bef

         # 그래프 그리기를 위한 x, y 값 생성
        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def make_lane(self, cv_image, white_fraction, yellow_fraction):
        """
        주어진 이미지에서 차선을 검출하고, 검출된 차선에 대한 레인(centerx)을 계산하는 함수.

        Args:
            cv_image: 검출할 이미지(numpy.ndarray).
            white_fraction: 검출된 하얀색 차선의 수직 방향 비율을 의미하는 변수.
            yellow_fraction: 검출된 노란색 차선의 수직 방향 비율을 의미하는 변수.

        Returns:
            None
        """
        # 검출된 차선 및 레인(centerx)을 그리기 위한 이미지 생성
        warp_zero = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), dtype=np.uint8)

        # 검출된 차선 및 레인을 그리기 위한 이미지 생성(검출된 차선의 색상이 다른 이미지)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # 검출된 차선 및 레인(centerx)을 그리기 위한 이미지 생성(검출된 차선의 색상이 다른 이미지)
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

        # y좌표에 대한 라인 좌표 생성
        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])

        # 검출된 노란색 차선이 300개 이상일 경우
        if yellow_fraction > 300:
            # 검출된 노란색 차선에 대한 좌표 계산
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            # 검출된 노란색 차선의 좌표로 라인 그리기
            cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)

        # 검출된 하얀색 차선이 300개 이상일 경우
        if white_fraction > 300:
            # 검출된 하얀색 차선에 대한 좌표 계산
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
             # 검출된 하얀색 차선의 좌표로 라인 그리기
            cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(255, 255, 0), thickness=25)

        # 레인(centerx)이 검출되었는지 여부 확인
        self.is_center_x_exist = True

        # 검출된 레인의 중심선(centerx)을 저장할 변수
        centerx = None

        # 하얀색 차선과 노란색 차선 신뢰도가 5 이상일 경우
        if self.reliability_white_line > 5 and self.reliability_yellow_line > 5:  
            # 하얀색 차선과 노란색 차선 모두 검출된 경우 
            if white_fraction > 300 and yellow_fraction > 300:
                # 노란색 차선과 하얀색 차선의 평균값을 centerx로 사용
                centerx = np.mean([self.left_fitx, self.right_fitx], axis=0)
                # 검출된 레인 중심선을 그림
                pts = np.hstack((pts_left, pts_right))
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
                cv2.polylines(color_warp_lines, np.int_([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)
                # 검출된 레인을 초록색으로 채움
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            # 하얀색 차선만 검출된 경우 
            if white_fraction > 300 and yellow_fraction <= 300:
                # 하얀색 차선의 평균값에서140을 빼준 값을 centerx로 사용
                centerx = np.subtract(self.right_fitx, 140)
                # 검출된 레인 중심선을 그림
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
                cv2.polylines(color_warp_lines, np.int_([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)
            # 노란색 차선만 검출된 경우 
            if white_fraction <= 300 and yellow_fraction > 300:
                 # 왼쪽 차선의 평균값에서 140을 더한 값을 centerx로 사용
                centerx = np.add(self.left_fitx, 140)
                # 검출된 레인 중심선을 그림
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
                cv2.polylines(color_warp_lines, np.int_([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)
        # 노란색 차선만 신뢰도가 5 이상일 경우
        elif self.reliability_white_line <= 5 and self.reliability_yellow_line > 5:
            # 노란색 차선의 오른쪽에 centerx 설정
            centerx = np.add(self.left_fitx, 140)
            # 검출된 레인 중심선 그리기
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(color_warp_lines, np.int_([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)
        # 하얀색 차선만 신뢰도가 5 이상일 경우
        elif self.reliability_white_line > 5 and self.reliability_yellow_line <= 5:
            # 하얀색 차선의 왼쪽에 centerx 설정
            centerx = np.subtract(self.right_fitx,140)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            # 검출된 레인 중심선 그리기
            cv2.polylines(color_warp_lines, np.int_([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)
        #만약 centerx 값이 없으면
        else:
            # is_center_x_exist 변수를 False로 설정하고 패스   
            self.is_center_x_exist = False
            pass

         # 원래 이미지(cv_image)와 검출된 레인 이미지(color_warp)를 합침
        final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
         # 레인 중심선 이미지(color_warp_lines)와 합침
        final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)
        # 이미지 크기를 2배로 변경
        final = cv2.resize(final, (0, 0), fx=2, fy=2)

        # centerx 값이 존재한다면
        if centerx is not None :
            if self.is_center_x_exist == True:

                # 레인 중심선을 구성하는 좌표값 중 y=120에 해당하는 x값을 msg_desired_center 변수에 저장
                msg_desired_center = Float64()
                msg_desired_center.data = centerx.item(120)

                #로봇 이동 함수 실행
                self.cbFollowLane(msg_desired_center)

        # centerx 값이 없다면  
        else: 
            #로봇 정지 함수 실행
            self.cmd_vel(0.0, 0.0)

        #self.pub_image_lane.publish(msg_desired_center)
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', final)[1]).tobytes()
        self.pub_image_lane.publish(msg)

        # 최종 이미지 보여주기
        cv2.imshow("final", final)
        # 이미지 유지 및 키 입력 대기
        cv2.waitKey(1)

    def cbFollowLane(self, desired_center):
        """
        주어진 레인 중심선(centerx)을 이용하여, 차량을 주행시키는 함수.

        Args:
        desired_center: 검출된 레인 중심선(centerx) 값이 포함된 메시지.

        Returns:
        None
        """
        #받아온 desired_center 을 center로 선언
        center = desired_center.data
        #오차계산
        error = center - 170.0
        # P,D 게인 설정
        Kp = 0.003
        Kd = 0.01
        # 오차의 변화율 계산
        error_diff = error - self.lastError
        #angular 값 계산
        angular_z = Kp * error + Kd * error_diff
        # 이전 오차 저장
        self.lastError = error
        # 속도와 조향값 계산
        twist = Twist()
        
        v = self.MAX_VEL * ((1.0 - abs(error) / 500) ** 2.2)
        linear = min(v, 0.12)

        angular = -angular_z
        
        # angular 값이 큰 경우 linear 값 1.2배 angular 0.8배
        if -angular_z < -0.13 or -angular_z > 0.13:
            linear = linear *1.2 
            angular = -angular_z*0.8
        
        # cmd_vel 메시지
        self.cmd_vel(linear, angular)

    def cmd_vel(self, linear, angular):
        """
        로봇의 움직임을 조정하는 함수.
        """
        twist = Twist()
        twist.linear.x = linear
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular
        self.pub_cmd_vel.publish(twist)

# 노드 객체를 생성하고, 노드를 실행하는 main 함수를 정의합니다.
def main(args=None):
    # ROS2 시스템을 초기화합니다.
    rclpy.init(args=args)
    # LineFollower 객체를 생성합니다.
    node = LineFollower()
    # 노드를 실행합니다.
    rclpy.spin(node)

if __name__ == '__main__':
    main()
