from PySide2 import QtCore
from rsData import log
from kafka import KafkaConsumer,KafkaProducer
from kafka.errors import KafkaError
from kafka import TopicPartition

class kfkProducer():
    # producer = None
    def __init__(self, broker, kafkaPort, kafkaTopic=''):
        self._broker = broker
        self._kafkaPort = kafkaPort
        self._kafkaTopic = kafkaTopic

    def __str__(self):
        log.info("--------------------------------")
        log.info("kafka-producer params ...")
        log.info("[KAFKA-BROKER]:%s" % self._broker)
        log.info("[KAFKA-PORT]:%s" % self._kafkaPort)
        log.info("[KAFKA-TOPIC]:%s" % self._kafkaTopic)
        log.info("--------------------------------")

    def registerKfkProducer(self):
        try:
            producer = KafkaProducer(bootstrap_servers = '{kafka_host}:{kafka_port}'.format(
                kafka_host=self._broker,
                kafka_port=self._kafkaPort
                ))
            return producer
        except KafkaError as e:
            log.info(e)
        return None

    def produceMsg(self, topic, msg, partition=0):
        # 自动将输入字符串转化为json格式，产出消息
        if(topic in ('', None)):
            log.error("topic is None, plz check!")
        else:
            try:
                # parmas_message = json.dumps(msg)#转化为json格式
                producer = self.registerKfkProducer()
                if producer:
                    producer.send(topic, value=msg, partition=partition)
                    producer.flush()
                # time.sleep(1)
            except KafkaError as e:
                log.info(e)

class Kafaka(QtCore.QThread):
    signalupdatePLCSig = QtCore.Signal()
    def __init__(self, parent=None):
        super(Kafaka, self).__init__(parent)
        self.working = True
        self.consumer = KafkaConsumer('jqr_tp_0608', bootstrap_servers=['127.0.0.1:9092'],consumer_timeout_ms=1000)

    def __del__(self):
        self.working = False

    def run(self):
        # consumer = KafkaConsumer('jqr_tp_0608', bootstrap_servers=['113.31.111.188:9092'])
        for msg in self.consumer:
            if(self.working == False):
                self.consumer.pause()
                break
            if(msg == {}):
                continue
            if msg.key == b'jqr':
                pass #kdata.jqr_msg = msg.value
                # self.signalupdatePLCSig.emit()
            else:
                pass