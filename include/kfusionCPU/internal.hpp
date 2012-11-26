#ifndef KFUSIONCPU_INTERNAL_H_
#define KFUSIONCPU_INTERNAL_H_

#include <opencv2/contrib/contrib.hpp>

namespace cvpr_tum
{

/******************\
 * HELPER STUFF *
 ******************/

template<typename T>
    bool is_nan(const T &value)
    {
        // True if NAN
        return value != value;
    }

/** \brief Camera intrinsics structure
 */
class Intr
{
public:
    float fx, fy, cx, cy;
    Intr()
    {
    }
    Intr(float fx_, float fy_, float cx_, float cy_) :
        fx(fx_), fy(fy_), cx(cx_), cy(cy_)
    {
    }

    Intr operator()(int level_index) const
    {
        int div = 1 << level_index;
        return (Intr(fx / div, fy / div, cx / div, cy / div));
    }

    friend inline std::ostream&
    operator <<(std::ostream& os, const Intr& intr)
    {
        os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
        return (os);
    }
};

struct ScopeTime
{
    const char* name;
    cv::TickMeter tm;
    ScopeTime(const char *name_) :
        name(name_)
    {
        tm.start();
    }
    ~ScopeTime()
    {
        tm.stop();
        std::cout << "Time(" << name << ") = " << tm.getTimeMilli() << "ms" << std::endl;
    }
};
}
#endif /* KFUSIONCPU_INTERNAL_H_ */
