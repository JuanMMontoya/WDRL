import tensorflow as tf

class DQN:
    def __init__(self, params, scope):
        self.scope = scope
        with tf.variable_scope(scope):
            self.x = tf.placeholder(tf.float32, shape=(None,params['width'],params['height'], params["mat_dim"]), name="X")
            self.q_t = tf.placeholder(tf.float32, shape=(None,), name="G")
            self.actions = tf.placeholder(tf.float32, shape=(None, params["k"]), name="actions")
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.terminals = tf.placeholder(tf.float32, [None], name='terminals')
            self.keep_prob = tf.placeholder(tf.float32)
            # calculate output and cost
            # calculate the size of the output of the final conv layer
            Z = self.x
            channels =params["mat_dim"]

            for filters, size, stride in params["conv_layer_sizes"]:
                w = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01))
                b = tf.Variable(tf.constant(0.1, shape=[filters]))
                c = tf.nn.conv2d(Z, w, strides=[1, stride, stride, 1], padding='SAME')
                Z = tf.nn.relu(tf.add(c, b))
                Z = tf.nn.dropout(Z, self.keep_prob)  #DROPOUT
                channels = filters

            # Fully connected layers

            Z = tf.contrib.layers.flatten(Z)
            _, dim = Z.shape
            dim = int(dim)
            for hidden in params["hidden_layer_sizes"]:
                w = tf.Variable(tf.random_normal([dim, hidden], stddev=0.01))
                b = tf.Variable(tf.constant(0.1, shape=[hidden]))
                mult = tf.add(tf.matmul(Z, w), b)
                Z = tf.nn.relu(mult)
                Z = tf.nn.dropout(Z, self.keep_prob) #Dropout
                dim = hidden

            # Final output layer
            self.w = tf.Variable(tf.random_normal([dim,params["k"]], stddev=0.01))
            b = tf.Variable(tf.constant(0.1, shape=[params["k"]]))
            self.y = tf.add(tf.matmul(Z, self.w), b)
            #Q,Cost,Optimizer
            self.discount = tf.constant(params['discount'])
            self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
            self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
            self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))

            if params['global_step'] is not None:
                self.global_step = tf.Variable(int(params['global_step']),name='global_step', trainable=False)
            else:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

           # self.optim = tf.train.RMSPropOptimizer(params['lr'],params['rms_decay'],0.0,params['rms_eps']).minimize(self.cost,global_step=self.global_step)
            self.optim = tf.train.AdamOptimizer(params['lr']).minimize(self.cost, global_step=self.global_step)

    def train(self,bat_s,bat_a,bat_t,q_t,bat_r, dropout=1):
        """
        Charge of training and calculating cost using Tf
        """
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r, self.keep_prob: dropout}
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost

    def set_session(self, sess):
        """
        Fix the Tensor Flow session
        :param session: Tf session object
        """
        self.sess = sess

    def predict(self, states, dropout = 1):
        """
        Makes predictions from images into q-values
        :param X: v-value with state
        :return: list with actions
        """
        pred = self.sess.run(self.y, feed_dict={self.x: states, self.keep_prob: dropout})
        return pred

    def rep_network(self, other):
        """
        Replace parameter of network with giving new Param.
        In charge of substitution between Q-Target Network and Q-Network
        :param other: new parameters from Target Network
        """
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        ops = []
        for p, q in zip(mine, theirs):
            actual = self.sess.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.sess.run(ops)

