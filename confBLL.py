import configparser

class ConfBLL():

    @staticmethod
    def readconf(section,key):
        conf = configparser.ConfigParser()
        conf.read("config.ini", encoding='utf8')
        return conf.get(section, key)

    @staticmethod
    def writeconf(section,key,value):
        conf = configparser.ConfigParser()
        conf.read("config.ini", encoding='utf8')
        conf.set(section,key,value)

        with open("config.ini", "w+") as f:
            conf.write(f)
            return True
        return False
