#ifndef SHAPE_H
#define SHAPE_H

double area_of(double side);

class Shape {
public:
    explicit Shape(double side);
    double area() const;

private:
    double side_;
};

#endif
