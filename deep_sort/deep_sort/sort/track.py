'''
Track类
轨迹类，保存一个track的状态信息
'''

class TrackState:

    Tentative = 1
    #不确定态，该状态在初始化一个track时分配
    #在连续n_init帧匹配上是转为confirmed态
    #没有匹配上任何detection时转为deleted态

    Confirmed = 2
    #确定态，表示track处于匹配状态
    #连续失配max_age次是转为deleted态

    Deleted = 3
    #删除态，表示track已失效


class Track:
'''
一个track的信息，包含(x,y,a,h) & v

其中：
    max_age：
    表示一个track的存活期限，默认70帧。与time_since_update比较
    当time_since_update超过max_age时，将该track从tracker泪飙中删除
    
    hits:
    表示连续确认的次数，用在Tentative态转为Confirmed态
    每次track进行update时，hits+1
    若hits>n_init（默认为3），即该track连续3帧都得到匹配，则Tentative态转为Confirmed态
        
    features:
    存储该track在不同帧对应位置通过ReID提取到的特征
    为了解决目标被遮挡后再次出现的问题，需要从以往帧对应的特征进行匹配
    特征过多会严重拖慢计算速度，故设置参数budget用来控制特征列表的长度，取最新的budget个features，将旧的删除掉
    
'''

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        #hits表示匹配次数，与n_init进行比较，超过n_init时设置为confirmed状态
        #hits每次update时（只有match才会update）进行一次更新
        self.age = 1 #同time_since_update
        self.time_since_update = 0 #每次调用predict函数时+1；每次调用update函数时置零

        self.state = TrackState.Tentative
        self.features = []
        
        #每个track对应多个feature，每次更新都将最新的feature添加到列表
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init #若连续n_init帧都没有失配，设置为deleted状态
        self._max_age = max_age

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
