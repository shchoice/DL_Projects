class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            # 해당 코드로 인해 싱글톤은 유지하지만 __init__() 함수가 있다면 그것을 수행하고 싱글톤 유지
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]