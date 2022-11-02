import torch


class TradingModel():

# u is action (alpha in original code) (vector)
# p is price (scalar)
# y is price distortion (scalar)
# x is inventory (vector)

  def __init__(self, dt, T_idx, N, initial, batch_size, control_model):
    self.T_idx = T_idx
    self.dt = dt
    self.N = N
    self.init = initial
    self.batch_size = batch_size
    self.control_model = control_model


  def inputs(self, t_idx, I_t, P_t, X_t, Y_t):
    return torch.cat((torch.broadcast_to(t_idx * self.dt, X_t.shape), \
                      torch.broadcast_to(I_t, X_t.shape), \
                      torch.broadcast_to(P_t, X_t.shape), \
                      X_t, \
                      torch.broadcast_to(Y_t, X_t.shape))).view(5, -1).t()
                      
  def control(self, t_idx, I_t, P_t, X_t, Y_t):
    N = self.N
    return self.control_model(self.inputs(t_idx, I_t, P_t, X_t, Y_t)).t().view(1, N, -1)


  def update_P(self, P, I):
    sigma = self.init[7]
    dt = self.dt
    batch_size = self.batch_size
    return P + \
              sigma * dt.sqrt() * torch.randn((1, batch_size), device=device, dtype=dtype) + \
              I * dt


  def update_I(self, I): # Ornstein–Uhlenbeck process
    beta = self.init[6]
    sigma = self.init[7]
    dt = self.dt
    batch_size = self.batch_size
    return I + \
              -beta * I * dt + \
              sigma * dt.sqrt() * torch.randn((1, batch_size), device=device, dtype=dtype)


  def update_X(self, X, u):     # Q: Individual strategies indexed by i?
    dt = self.dt
    return X + \
              -u * dt     # A: For other stuff, yes. Here vector op works.


  def update_Y(self, Y, u):    # Y is a scalar (not considering batch size)
    rho = self.init[3]
    gamma = self.init[4]
    N = self.N
    dt = self.dt
    return Y + \
              -rho * Y * dt + \
              gamma * u.mean(axis=1).detach() * dt
              # ((1/N)*u.mean(axis=1) + (1-1/N)*u.mean(axis=1).detach()) * dt # detach so MFG and not MFC

              
  def running_cost(self, P, X, Y, u):
    lambd = self.init[0]
    kappa = self.init[1]
    phi = self.init[2]
    dt = self.dt
    return -(P - kappa * Y) * u * dt + \
              lambd * u.square() * dt + \
              phi * X.square() * dt


  def terminal_cost(self, P_T, X_T, Y_T):
    # terminal cost was changed in update to paper
    kappa = self.init[1]
    fancy_rho = self.init[5]
    return -X_T * (P_T - kappa * Y_T) + \
              fancy_rho * X_T.square()

              
  def sample(self):
    _ , _ , _ , _ , _ , _ , _ , _ , p_init, y_init, i_init = self.init
    N = self.N
    batch_size = self.batch_size
    T_idx = self.T_idx

    #for deterministic
    n = int(np.ceil(N/2))
    x_init = np.concatenate((np.linspace(10-n,9,num=n), np.linspace(11,10+n,num=n))).reshape(1, N, batch_size)

    P = p_init * torch.ones((1, batch_size), device=device, dtype=dtype)
    # X = torch.randn((1, N, batch_size), device=device, dtype=dtype) + 10 # Note the first coordinate is time, it seems necessary to use
    X = torch.tensor(x_init, device=device, dtype=dtype) # for deterministic
    Y = y_init * torch.ones((1, batch_size), device=device, dtype=dtype)
    I = i_init * torch.ones((1, batch_size), device=device, dtype=dtype)

    for t_idx in range(T_idx):    # order of update probably matters, but idk how yet
      I = torch.cat((I,
            self.update_I(I[t_idx])), 0)

      P = torch.cat((P,
            self.update_P(P[t_idx], I[t_idx])), 0)

      X = torch.cat((X, 
            self.update_X(X[t_idx], self.control(t_idx, I[t_idx], P[t_idx], X[t_idx], Y[t_idx]))), 0)
          
      Y = torch.cat((Y,
            self.update_Y(Y[t_idx], self.control(t_idx, I[t_idx], P[t_idx], X[t_idx], Y[t_idx]))), 0)

    return I, P, X, Y
    
    
    def sample_from_signal(self, I, P):
      _ , _ , _ , _ , _ , _ , _ , _ , _ , y_init, _ = self.init
      N = self.N
      batch_size = self.batch_size
      T_idx = self.T_idx
      
      #for deterministic
      n = int(np.ceil(N/2))
      x_init = np.concatenate((np.linspace(10-n,9,num=n), np.linspace(11,10+n,num=n))).reshape(1, N, batch_size)

      # X = torch.randn((1, N, batch_size), device=device, dtype=dtype) + 10 # Note the first coordinate is time, it seems necessary to use
      X = torch.tensor(x_init, device=device, dtype=dtype) # for deterministic
      # X = 10 * torch.ones((1, N, batch_size), device=device, dtype=dtype)
      Y = y_init * torch.ones((1, batch_size), device=device, dtype=dtype)

      for t_idx in range(T_idx):    # order of update probably matters, but idk how yet
        X = torch.cat((X, 
              self.update_X(X[t_idx], self.control(t_idx, I[t_idx], P[t_idx], X[t_idx], Y[t_idx]))), 0)
            
        Y = torch.cat((Y,
              self.update_Y(Y[t_idx], self.control(t_idx, I[t_idx], P[t_idx], X[t_idx], Y[t_idx]))), 0)

      return X, Y

  def cost(self):
    I, P, X, Y = self.sample()
    l = torch.zeros_like(X[0])
    T_idx = self.T_idx

    # this causes a computational bottleneck: N is done in a parallel way

    for t_idx in range(T_idx):

      l = l + self.running_cost(P[t_idx], X[t_idx], Y[t_idx], self.control(t_idx, I[t_idx], P[t_idx], X[t_idx], Y[t_idx]))

    l = l + self.terminal_cost(P[T_idx], X[T_idx], Y[T_idx])

    return l.mean()

class sample_signal():

  def __init__(self, dt, T_idx, initial, batch_size):
    self.T_idx = T_idx
    self.dt = dt
    self.init = initial
    self.batch_size = batch_size

  def update_P(self, P, I):
    sigma = self.init[7]
    dt = self.dt
    batch_size = self.batch_size
    return P + \
              sigma * dt.sqrt() * torch.randn((1, batch_size), device=device, dtype=dtype) + \
              I * dt

  def update_I(self, I): # Ornstein–Uhlenbeck process
    beta = self.init[6]
    sigma = self.init[7]
    dt = self.dt
    batch_size = self.batch_size
    return I + \
              -beta * I * dt + \
              sigma * dt.sqrt() * torch.randn((1, batch_size), device=device, dtype=dtype)

  def signal(self):
    _ , _ , _ , _ , _ , _ , _ , _ , p_init, y_init, i_init = self.init
    batch_size = self.batch_size

    P = p_init * torch.ones((1, batch_size), device=device, dtype=dtype)
    I = i_init * torch.ones((1, batch_size), device=device, dtype=dtype)

    T_idx = self.T_idx

    for t_idx in range(T_idx):    # order of update probably matters, but idk how yet
      I = torch.cat((I,
            self.update_I(I[t_idx])), 0)

      P = torch.cat((P,
            self.update_P(P[t_idx], I[t_idx])), 0)
      
    return I, P