package com.example.app

import com.example.util.Formatter

fun run(name: String): String {
    val formatter = Formatter()
    return formatter.format(name)
}
