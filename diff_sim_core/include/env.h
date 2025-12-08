// include/env.h

class Env {
public:
    virtual ~Env() = default;

    // Returns: Initial Observation (Tensor)
    virtual Tensor reset() = 0;

    // Input: Action (Tensor)
    // Returns: Tuple <Observation, Reward, Done, Info>
    // Info is a string for now (e.g. "goal_reached")
    virtual std::tuple<Tensor, float, bool, std::string> step(const Tensor& action) = 0;

    virtual void render() {}
};