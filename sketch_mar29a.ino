
int piezoPin = 6;
void setup() {
  Serial.begin(9600); // 시리얼 통신을 9600 bps로 초기화합니다.
  pinMode(piezoPin, OUTPUT);
}
int flag = 0;
void loop() {
  if (Serial.available() > 0) { // 시리얼 버퍼에 데이터가 있는지 확인합니다.
    int receivedValue = Serial.parseInt(); // 시리얼에서 정수 값을 읽어옵니다.
    if(receivedValue == 1){
      flag = 1;
    }
    else if(receivedValue == 2){
      flag =2;
    }
    else if(receivedValue == 3){
      flag = 0;
    }
    Serial.println("Received: " + String(receivedValue)); // 읽어온 값을 시리얼로 다시 보냅니다.
  }
  if(flag == 1 || flag == 2){ // flag가 1 또는 2일 때 부저를 울립니다.
    for(int freq = 150; freq <=1800; freq = freq + 2) {
        tone(piezoPin, freq, 10);
      }

    for(int freq = 1800; freq <=150; freq = freq - 2) {
      tone(piezoPin, freq, 10);
    }
  }
  else if(flag == 0){ // flag가 0일 때 부저를 멈춥니다.
    digitalWrite(piezoPin,LOW);
  }
}
