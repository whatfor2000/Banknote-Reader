# Banknote-Reader
CNNs Gym
  ในรีโพซิโทรี่นี้เป็นเครื่องมือพร้อมคู่มือการใช้งาน สำหรับการสร้าง Machine Learning Model ทั้ง TFlite เเละ Keras Format

  โมเดล Machine Learning ที่เหมาะสมสำหรับงานนี้คือ Convolutional Neural Network (CNN) ซึ่งเป็นโมเดลที่ได้รับความนิยมในการประมวลผลภาพ โมเดลนี้จะถูกฝึกด้วยชุดข้อมูลที่มีภาพธนบัตรและป้ายกำกับที่ตรงกัน การฝึกนี้จะช่วยให้โมเดลสามารถรู้จำและแยกแยะธนบัตรต่าง ๆ ในชุดข้อมูลทดสอบหลังจากฝึกโมเดลเสร็จสิ้น จำเป็นต้องทดสอบด้วยชุดข้อมูลที่ไม่เคยให้โมเดลเห็นมาก่อน เพื่อประเมินประสิทธิภาพของโมเดล เมื่อโมเดลถูกฝึกเรียบร้อยและได้รับการประเมินอย่างมีประสิทธิภาพ นำโมเดลนี้มาใช้ในระบบเครื่องอ่านธนบัตรเครื่องอ่านธนบัตรที่ใช้เทคนิคการเรียนรู้ของเครื่องเป็นเครื่องมือที่มีความสามารถในการจำแนกประเภทของธนบัตรอัตโนมัติ

วิธีการใช้งาน
1.เตรียมข้อมูลสำหรับการฝึกเอไอ โดยเเบ่งโฟรเดอร์เเยกกันเเต่ละคลาส ในโฟรเดอร์ training ,validation และ Sample<br>
2.เปิดไฟล์ training.py เเล้วปรับปรุงตรวจสอบสถาปัตยกรรมนิวรอนเน็ตเวิร์คให้ได้ตามต้องการ<br>
3.เปิดไฟล์ Main.py เเล้วปรับปรุงตรวจสอบคุณสมบัติต่างๆ<br>
4.Run Main.py เพื่อเริ่มการทำงาน<br><br>

***โปรเเกรมนี้ไม่มีการเช็คพ้อยหรือทำต่อจากจุดเช็คพ้อย<br>
Dev by Predpa/whatfor2000<br>
