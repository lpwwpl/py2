import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.0
import QtWebEngine 1.0
import QtWebChannel 1.0

Rectangle {
    id: root
    visible: true

    WebEngineView {
        id: webview
        anchors.fill: parent
        url: myUrl
        onLoadingChanged: {
            switch (loadRequest.status) {
            case WebEngineLoadRequest.LoadStartedStatus:
                loadProgressBar.visible = true
                break
            default:
                loadProgressBar.visible = false
                break
            }
        }
        onLoadProgressChanged: loadProgressBar.value = loadProgress
    }

    ProgressBar {
        id: loadProgressBar
        width: root.width
        from: 0
        to: 100
        value: 0
        visible: false
    }
}