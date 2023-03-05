import paho.mqtt.client as mqtt
class mqttHandle(object):

    def __init__(self,mqtt_info):
        self.mqtt_info=mqtt_info

    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("chat")

    def on_message(client, userdata, msg):
        print("topic:" + msg.topic + " payload:" + str(msg.payload))

    def publish(self):
        client = mqtt.Client()
        client.on_connect = mqttHandle.on_connect
        client.on_message = mqttHandle.on_message
        # client.username_pw_set(self.mqtt_info['username'], self.mqtt_info['password'])
        client.connect(self.mqtt_info['host'], self.mqtt_info['port'], 60)
        client.publish(self.mqtt_info['topic'], str(self.mqtt_info['payload']))
        #client.loop_forever()
        client.disconnect()
        print('publish topic over')

if __name__=="__main__":
    mqtt_info={
        'username':'admin',
        'password':'password',
        'host':'127.0.0.1',
        'port':61613,
        'topic':'joints',
        'payload':'hello world',
        }
    mqtt_info="2.03, 35.28, -8.94, -1.11, 63.70, 90.94"
    mqttc=mqttHandle(mqtt_info)
    mqttc.publish()



