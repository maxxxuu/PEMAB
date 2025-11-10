class SingleResult:
    def __init__(self, win=False, total_reward=0, env_snap=None):
        self.win = win
        self.total_reward = total_reward
        self.play_history = None
        self.q_values = None
        self.env_snap = env_snap
        self.states = None

    def get_total_reward(self):
        return self.total_reward

    def get_result(self):
        return self.win, self.total_reward, self.play_history, self.q_values, self.env_snap

    def get_result_dict(self):
        return vars(self)

    def update_reward(self, new_reward):
        self.total_reward += new_reward

    def update_history(self, play_history):
        self.play_history = play_history

    def update_q_values(self, q_values):
        self.q_values = q_values

    def update_env_snap(self, env_snap):
        self.env_snap = env_snap

    def update_states(self, states):
        self.states = states

    def judged_win(self):
        self.win = True


