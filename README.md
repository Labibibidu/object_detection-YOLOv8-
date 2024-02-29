# object_detection-YOLOv8
此專案使用YOLOv8對car.mp4影片中的車輛進行辨識，並設計一個counter紀錄車子的數量

# 整體大致流程:
1.先使用mask將圖像做遮罩，只留下想要辨識的區域。
2.用YOLOv8模型對辨識區域做辨識，得到辨識結果(包含box，類別id等等)
3.用cvzone將車子的框，id繪製出來
4.設定與繪製紅線作為判定車子計數的基準線
5.計算車子的中心點，若車子中心點經過了紅線，則紅線變為綠線，並車輛總數顯示+1

# 參考資料:
[1]https://www.youtube.com/watch?v=WgPbbWmnXJ8&ab_channel=Murtaza'sWorkshop-RoboticsandAI
[2]https://www.youtube.com/watch?v=Zd5jSDRjWfA&t=6s&ab_channel=走歪的工程師James

