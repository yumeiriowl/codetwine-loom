package com.example.app;

import com.example.util.Greeter;

public class App {
    public static String run(String name) {
        Greeter greeter = new Greeter();
        return greeter.greet(name);
    }
}
