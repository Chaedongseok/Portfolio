#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <softPwm.h>

//Set the GPIO pinnumber
#define Power_tilt ??
#define Direction_tilt ??
#define Direction_tiltd ??

void Setup(); //A function that declares a GPIO variable
void SpinMotor(int pin, int frequency, double pos); //It is a function to rotate the motor


int state_end = 20;
int frequency = 50;

int main(int argc, char *argv[])
{
	int state_num[state_end];


 	Setup();
        digitalWrite(Power_tilt,LOW);
        digitalWrite(Direction_tiltd,LOW);
	digitalWrite(Direction_tilt,LOW);

	if(wiringPiSetup() == -1){
		exit(1);
	}
	double pos_tilt =80;
//	int frequency = 50;
	double change_size_tilt =1;
	for(int i = 0; i<10;i++){
		SpinMotor(Power_tilt, frequency, pos_tilt);
	}
	while(1){
//		Setup();					
//		delayMicroseconds((1000/frequency)*100);
//		delay(20);
		//Accept direction and rotate motor
		if(digitalRead(Direction_tilt) == 1 && digitalRead(Direction_tiltd == 1))continue;
                digitalWrite(Direction_tiltd,LOW);
                digitalWrite(Direction_tilt,LOW);

		state_num[state_end-1] =0;
		if(digitalRead(Direction_tilt)==1 && digitalRead(Direction_tiltd)==0){ // UP
			pos_tilt -= change_size_tilt;
//            		printf("Up \n");
//			printf("%f \n", pos_tilt);
			state_num[state_end-1]++;
		}
		if(digitalRead(Direction_tiltd)==1 && digitalRead(Direction_tilt)==0){ // Down
			pos_tilt += change_size_tilt;
//            		printf("Down \n");
//			printf("%f \n", pos_tilt);
			state_num[state_end-1]++;
		}

         	for(int i= 0; i<state_end-1;i++){
			state_num[i] = state_num[i+1];
		}
 

		int state_sum = 0;
		for(int i = 0; i<state_end; i++){
			state_sum += state_num[i]; 
		}
		//stop motor
        if(state_sum ==0){
			pinMode(Power_tilt, OUTPUT);
			digitalWrite(Power_tilt, LOW);
//			printf("motor stop");
//			Setup();
			delay(10);
			
			continue;
			
		}

		state_num[state_end] = 0;
		if(pos_tilt>= 200){
			pos_tilt = 200;
		}
		if(pos_tilt<= 50)
			pos_tilt = 50;
		
		SpinMotor(Power_tilt, frequency, pos_tilt);
//		printf("%f \n",pos_tilt);
	}
	return 0;
}
void Setup()
{
	pinMode(Power_tilt,OUTPUT);
	pinMode(Direction_tiltd,INPUT);
	pinMode(Direction_tilt,INPUT); 
}

void SpinMotor(int pin, int frequency, double pos)//Create a PWM signal
{
        digitalWrite(pin, 1);
        delayMicroseconds(10*pos);
        digitalWrite(pin, 0);
        delayMicroseconds(((1000/frequency)*1000-10*pos));
	//delay(20);
}

