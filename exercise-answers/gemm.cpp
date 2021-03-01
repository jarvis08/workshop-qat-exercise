void cal_qsz(const float _min, const float _max, float *QS, int8_t *QZ)
{
    const float qmin = -128;
    const float qmax = 127;

    float scale = (_max - _min) / (qmax - qmin);
    float initial_zero_point = qmin - (_min / scale);

    std::int8_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) nudged_zero_point = qmin;
    else if (initial_zero_point > qmax) nudged_zero_point = qmax;
    else nudged_zero_point = static_cast<std::int8_t>(std::round(initial_zero_point));

    *QS = scale;
    *QZ = nudged_zero_point;
}
