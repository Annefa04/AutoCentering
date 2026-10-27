#pragma warning(disable : 4996)

#include <QApplication>

#include "MainWindow.h"
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    AutoCentering window;
    window.show();
    return app.exec();
}
