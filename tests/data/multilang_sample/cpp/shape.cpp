#include "shape.h"

double area_of(double side) {
    return side * side;
}

Shape::Shape(double side) : side_(side) {}

double Shape::area() const {
    return area_of(side_);
}
