"""
This module contains the file system tree view.
"""
import locale
import os
import platform
import shutil
import subprocess
import rsData
from PySide2 import QtCore, QtGui, QtWidgets
import sys
def icon(theme_name='', path='', qta_name='', qta_options=None, use_qta=None):
    """
    Creates an icon from qtawesome, from theme or from path.

    :param theme_name: icon name in the current theme (GNU/Linux only)
    :param path: path of the icon (from file system or qrc)
    :returns: QtGui.QIcon
    """
    ret_val = None
    if theme_name and path:
        ret_val = QtGui.QIcon.fromTheme(theme_name, QtGui.QIcon(path))
    elif theme_name:
        ret_val = QtGui.QIcon.fromTheme(theme_name)
    elif path:
        ret_val = QtGui.QIcon(path)
    return ret_val


class FileIconProvider(QtWidgets.QFileIconProvider):
    def icon(self, type_or_info):
        if isinstance(type_or_info, QtCore.QFileInfo):
            if 'linux' not in sys.platform:
                # use hardcoded icon on windows/OSX
                if type_or_info.isDir():
                    return QtGui.QIcon('./Resource/images/folder.png')
                else:
                    return QtGui.QIcon('./Resource/images/file.png')
        else:
            if 'linux' not in sys.platform:
                if type_or_info == self.Folder:
                    return QtGui.QIcon('./Resource/images/file.png')
                elif type_or_info == self.File:
                    return QtGui.QIcon('')
        return super().icon(type_or_info)

class FileSystemWidget(QtWidgets.QWidget):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.initUI()


    def set_root_path(self,path):
        self.tree.set_root_path(path)

    def root_path_slot(self,path):
        font = self.text.font()
        fm = QtGui.QFontMetrics(font)

        self.text.setText(fm.elidedText(path,QtCore.Qt.ElideRight,self.text.width()))
        self.text.setToolTip(path)

    def initUI(self):
        self.tree = FileSystemTreeView()
        self.text=QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        self.text.setVerticalScrollBarPolicy(QtGui.Qt.ScrollBarAlwaysOff)
        font = QtGui.QFont("宋体", 10, QtGui.QFont.Bold)
        fm = QtGui.QFontMetrics(font)
        self.text.setFixedHeight(fm.height())
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.tree)
        self.layout.addWidget(self.text)
        self.setLayout(self.layout)
        self.tree.root_path_signal.connect(self.root_path_slot)
        # self.tree.set_root_path("")

        self.mnu = FSContextMenu(self)
        self.tree.set_context_menu(self.mnu)

        # self.textEdit.document().setMaximumBlockCount(100)
        # self.textEdit.setVerticalScrollBarPolicy(QtGui.Qt.ScrollBarAsNeeded)

class FileSystemTreeView(QtWidgets.QTreeView):
    """
    Extends QtWidgets.QTreeView with a filterable file system model.

    To exclude directories or extension, just set
    :attr:`FilterProxyModel.ignored_directories` and
    :attr:`FilterProxyModel.ignored_extensions`.

    Provides methods to retrieve file info from model index.

    By default there is no context menu and no file operations are possible. We
    provide a standard context menu with basic file system operation (
    :class:`FileSystemContextMenu`) that you can extend and
    set on the tree view using :meth:`FileSystemTreeView.set_context_menu`

    """
    class FilterProxyModel(QtCore.QSortFilterProxyModel):
        """
        Excludes :attr:`ignored_directories` and :attr:`ignored_extensions`
        from the file system model.
        """
        def __init__(self):
            super(FileSystemTreeView.FilterProxyModel, self).__init__()
            self.patterns = [
                'runs', 'cali*', 'pics*']
            self.root = None
            self.parent_dir = None

        def set_root_path(self, path):
            """
            Sets the root path to watch.
            :param path: root path (str).
            """
            self.root = path
            self.parent_dir = os.path.dirname(self.root)



        def rootDirectory(self):
            return self.sourceModel().rootDirectory()

        # def filterAcceptsRow(self, row, parent):
        #     index0 = self.sourceModel().index(row, 0, parent)
        #     finfo = self.sourceModel().fileInfo(index0)
        #
        #
        #     if finfo.path() == self.parent_dir:
        #         return True
        #     if finfo.path() == self.root:
        #         return True
        #         # for item in os.listdir(self.root):
        #         #     for ptrn in self.patterns:
        #         #         print(item)
        #         #         if fnmatch.fnmatch(item, ptrn):
        #         #
        #         #             return True
        #         #         else:
        #         #             return False
        #
        #
        #     return False


            # if finfo
            # index = self.sourceModel().index(row, 0, index0)
            # for i in range(self.sourceModel().rowCount(index0)):
            #     if self.filterAcceptsRow(i, index):
            #         return True

            # index0 = self.sourceModel().index(row, 0, parent)
            # finfo = self.sourceModel().fileInfo(index0)
            # fn = finfo.fileName()
            # print(row)
            # for ptrn in self.patterns:
            #     if fnmatch.fnmatch(fn, ptrn):
            #         return True
            #     else:
            #         return False
            return True

    #: signal emitted when the user deleted a file or a directory
    #: Deprecated, use files_deleted instead.
    #: Parameters:
    #: - path (str): path of the file that got deleted
    #: Note that if the removed path is a directory, this signal will be emitted for every file
    #: found recursively in the parent directory
    file_deleted = QtCore.Signal(str)
    root_path_signal = QtCore.Signal(str)
    #: Signal emitted when the user deleted a file or a directory,
    #: it is emitted only once with all the files deleted.
    files_deleted = QtCore.Signal(list)

    #: signal emitted when the user renamed a file or a directory
    #: Deprecated, use files_renamed instead.
    #: Parameters:
    #: - old (str): old path
    #: - new (str): new path
    file_renamed = QtCore.Signal(str, str)

    #: Signal emitted when the user renamed a file or a directory,
    #: it is emitted once with all the renamed files (not directgories)
    files_renamed = QtCore.Signal(list)

    #: signal emitted when the user created a file
    #: Parameters:
    #: - path (str): path of the file that got created
    file_created = QtCore.Signal(str)

    #: signal emitted just before the context menu is shown
    #: Parameters:
    #:   - file path: current file path.
    about_to_show_context_menu = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(FileSystemTreeView, self).__init__(parent)
        self._path_to_set = None
        self._path_to_select = None
        self.context_menu = None
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.helper = FileSystemHelper(self)
        self.setSelectionMode(self.ExtendedSelection)
        self._icon_provider = FileIconProvider()
        self._hide_extra_colums = True

    def _on_set_as_root_triggered(self):

        path = self.helper.get_current_path()
        if not path.startswith(rsData.root_path):
            return
        if os.path.isfile(path):
            return
        self.set_root_path(path)

    def mouseDoubleClickEvent(self, event):
        self._on_set_as_root_triggered()

    def showEvent(self, event):
        super(FileSystemTreeView, self).showEvent(event)
        if self._path_to_set:
            self.set_root_path(self._path_to_set, self._hide_extra_colums)
            self._path_to_set = None

    def set_icon_provider(self, icon_provider):
        self._icon_provider = icon_provider

    def set_context_menu(self, context_menu):
        """
        Sets the context menu of the tree view.

        :param context_menu: QMenu
        """
        self.context_menu = context_menu
        self.context_menu.tree_view = self
        self.context_menu.init_actions()
        for action in self.context_menu.actions():
            self.addAction(action)

    def set_root_path(self, path, hide_extra_columns=True):
        """
        Sets the root path to watch
        :param path: root path - str
        :param hide_extra_columns: Hide extra column (size, paths,...)
        """
        self._hide_extra_colums = hide_extra_columns

        if os.path.isfile(path):
            path = os.path.abspath(os.path.join(path, os.pardir))
        self._fs_model_source = QtWidgets.QFileSystemModel()
        self._fs_model_source.setFilter(QtCore.QDir.Dirs | QtCore.QDir.Files | QtCore.QDir.NoDot |
                                        QtCore.QDir.Hidden)
        self._fs_model_source.setIconProvider(self._icon_provider)
        self._fs_model_proxy = self.FilterProxyModel()
        self._fs_model_proxy.setSourceModel(self._fs_model_source)
        rsData.cur_root_path = path

        self._fs_model_source.directoryLoaded.connect(self._on_path_loaded)
        self.setModel(self._fs_model_proxy)
        self._fs_model_proxy.set_root_path(path)
        self._on_path_loaded(rsData.cur_root_path)
        self.root_path_signal.emit(rsData.cur_root_path)

    def _on_path_loaded(self, path):
        try:
            self.setModel(self._fs_model_proxy)
            file_root_index = self._fs_model_source.setRootPath(rsData.cur_root_path)
            root_index = self._fs_model_proxy.mapFromSource(file_root_index)
            self.setRootIndex(root_index)
            if not os.path.ismount(rsData.cur_root_path):
                self.expandToDepth(0)
            if self._hide_extra_colums:
                self.setHeaderHidden(True)
                for i in range(1, 4):
                    self.hideColumn(i)
            if self._path_to_select:
                self.select_path(self._path_to_select)
                self._path_to_select = None
        except RuntimeError:
            # wrapped C/C++ object of type FileSystemTreeView has been deleted
            return

    def filePath(self, index):
        return self._fs_model_source.filePath(
            self._fs_model_proxy.mapToSource(index))

    def fileInfo(self, index):
        return self._fs_model_source.fileInfo(
            self._fs_model_proxy.mapToSource(index))

    def _show_context_menu(self, point):
        if self.context_menu:
            self.about_to_show_context_menu.emit(
                FileSystemHelper(self).get_current_path())
            self.context_menu.exec_(self.mapToGlobal(point))

    def select_path(self, path):
        if not self.isVisible():
            self._path_to_select = path
        else:
            self.setCurrentIndex(self._fs_model_proxy.mapFromSource(
                self._fs_model_source.index(path)))


class FileSystemHelper:
    """
    File system helper. Helps manipulating the clipboard for file operations
    on the tree view (drag & drop, context menu, ...).
    """
    class _UrlListMimeData(QtCore.QMimeData):
        def __init__(self, copy=True):
            super(FileSystemHelper._UrlListMimeData, self).__init__()
            self.copy = copy

        def set_list(self, urls):
            """
            Sets the lis of urls into the mime type data.
            """
            lst = []
            for url in urls:
                lst.append(bytes(url, encoding=locale.getpreferredencoding()))
            self.setData(self.format(self.copy), b'\n'.join(lst))

        @classmethod
        def list_from(cls, mime_data, copy=True):
            """
            Returns a list of url from mimetype data
            :param mime_data: mime data from which we must read the list of
                urls
            :param copy: True to copy, False to cut
            """
            string = bytes(mime_data.data(cls.format(copy))).decode('utf-8')
            lst = string.split('\n')
            urls = []
            for val in lst:
                urls.append(val)
            return urls

        def formats(self):
            return [self.format(self.copy)]

        @classmethod
        def format(cls, copy=True):
            return 'text/tv-copy-url-list' if copy else 'text/tv-cut-url-list'

    def __init__(self, treeview):
        self.tree_view = treeview

    def selected_urls(self):
        """
        Gets the list of selected items file path (url)
        """
        urls = []
        for proxy_index in self.tree_view.selectedIndexes():
            finfo = self.tree_view.fileInfo(proxy_index)
            urls.append(finfo.canonicalFilePath())
        return urls

    @staticmethod
    def _get_files(path):
        """
        Returns the list of files contained in path (recursively).
        """
        ret_val = []
        for root, _, files in os.walk(path):
            for f in files:
                ret_val.append(os.path.join(root, f))
        return ret_val

    def delete(self):
        """
        Deletes the selected items.
        """
        urls = self.selected_urls()
        rep = QtWidgets.QMessageBox.question(
            self.tree_view, ('Confirm delete'),
            ('Are you sure about deleting the selected files/directories?'),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes)
        if rep == QtWidgets.QMessageBox.Yes:
            deleted_files = []
            for fn in urls:
                try:
                    if os.path.isfile(fn):
                        os.remove(fn)
                        deleted_files.append(fn)
                    else:
                        files = self._get_files(fn)
                        shutil.rmtree(fn)
                        deleted_files += files
                except OSError as e:
                    QtWidgets.QMessageBox.warning(
                        self.tree_view, ('Delete failed'),
                        ('Failed to remove "%s".\n\n%s') % (fn, str(e)))
            self.tree_view.files_deleted.emit(deleted_files)
            for d in deleted_files:
                self.tree_view.file_deleted.emit(os.path.normpath(d))

    def get_current_path(self):
        """
        Gets the path of the currently selected item.
        """
        var = self.tree_view.fileInfo(
            self.tree_view.currentIndex())

        path = var.absoluteFilePath()
        # https://github.com/pyQode/pyQode/issues/6
        if not path:
            path = rsData.cur_root_path
        return path

    def copy_path_to_clipboard(self):
        """
        Copies the file path to the clipboard
        """
        path = self.get_current_path()
        QtWidgets.QApplication.clipboard().setText(path)

    def rename(self):
        """
        Renames the selected item in the tree view
        """
        src = self.get_current_path()
        pardir, name = os.path.split(src)
        new_name, status = QtWidgets.QInputDialog.getText(
            self.tree_view, ('Rename '), ('New name:'),
            QtWidgets.QLineEdit.Normal, name)
        if status:
            dest = os.path.join(pardir, new_name)
            old_files = []
            if os.path.isdir(src):
                old_files = self._get_files(src)
            else:
                old_files = [src]
            try:
                os.rename(src, dest)
            except OSError as e:
                QtWidgets.QMessageBox.warning(
                    self.tree_view, ('Rename failed'),
                    ('Failed to rename "%s" into "%s".\n\n%s') % (src, dest, str(e)))
            else:
                if os.path.isdir(dest):
                    new_files = self._get_files(dest)
                else:
                    new_files = [dest]
                self.tree_view.file_renamed.emit(os.path.normpath(src),
                                                 os.path.normpath(dest))
                renamed_files = []
                for old_f, new_f in zip(old_files, new_files):
                    self.tree_view.file_renamed.emit(old_f, new_f)
                    renamed_files.append((old_f, new_f))
                # emit all changes in one go
                self.tree_view.files_renamed.emit(renamed_files)

    def create_directory(self):
        """
        Creates a directory under the selected directory (if the selected item
        is a file, the parent directory is used).
        """
        src = self.get_current_path()
        name, status = QtWidgets.QInputDialog.getText(
            self.tree_view, ('Create directory'), ('Name:'),
            QtWidgets.QLineEdit.Normal, '')
        if status:
            fatal_names = ['.', '..']
            for i in fatal_names:
                if i == name:
                    QtWidgets.QMessageBox.critical(
                        self.tree_view, ("Error"), ("Wrong directory name"))
                    return

            if os.path.isfile(src):
                src = os.path.dirname(src)
            dir_name = os.path.join(src, name)
            try:
                os.makedirs(dir_name, exist_ok=True)
            except OSError as e:
                QtWidgets.QMessageBox.warning(
                    self.tree_view, ('Failed to create directory'),
                    ('Failed to create directory: "%s".\n\n%s') % (dir_name, str(e)))


class FileSystemContextMenu(QtWidgets.QMenu):
    """
    Default context menu for the file system treeview.

    This context menu contains the following actions:
        - Copy
        - Cut
        - Paste
        - Delete
        - Copy path

    .. note:: copy/cut/paste action works only from inside the application
        (e.g. you cannot paste what you copied in the app to the explorer)

    """
    _explorer = None
    _command = None

    def __init__(self):
        super(FileSystemContextMenu, self).__init__()
        #: Reference to the tree view
        self.tree_view = None

    def addAction(self, *args):
        action = super(FileSystemContextMenu, self).addAction(*args)
        if action is None:
            action = args[0]
        action.setShortcutContext(QtCore.Qt.WidgetShortcut)
        self.tree_view.addAction(action)
        return action

    def init_actions(self):
        # New - submenu
        self.menu_new = self.addMenu("&New")
        self.menu_new.setIcon(
            icon('document-new', None, 'fa.plus'))
        # https://github.com/pyQode/pyqode.core/pull/153
        # new_user_actions = self.get_new_user_actions()
        # if len(new_user_actions) > 0:
        #     self.menu_new.addSeparator()
        #     for user_new_action in new_user_actions:
        #         self.menu_new.addAction(user_new_action)
        # New file
        # self.action_create_file = QtWidgets.QAction(('&File'), self)
        # self.action_create_file.triggered.connect(
        #     self._on_create_file_triggered)
        icon_provider = self.tree_view._icon_provider
        # self.action_create_file.setIcon(icon_provider.icon(
        #     icon_provider.File))
        # self.menu_new.addAction(self.action_create_file)
        # New directory
        self.action_create_directory = QtWidgets.QAction(
            ('&Directory'), self)
        self.action_create_directory.triggered.connect(
            self._on_create_directory_triggered)
        self.action_create_directory.setIcon(icon_provider.icon(
            icon_provider.Folder))
        self.menu_new.addAction(self.action_create_directory)
        self.addSeparator()

        self.action_set_as_root = QtWidgets.QAction('SetAs Root',self)
        # self.action_set_as_root.setShortcut(QtGui.QKeySequence.)
        self.addAction(self.action_set_as_root)
        self.action_set_as_root.triggered.connect(self._on_set_as_root_triggered)

        # copy path
        self.action_copy_path = QtWidgets.QAction(('&Copy path'), self)
        self.action_copy_path.setShortcut('Ctrl+Shift+C')
        self.addAction(self.action_copy_path)
        self.action_copy_path.triggered.connect(self._on_copy_path_triggered)
        # Rename
        self.action_rename = QtWidgets.QAction(('&Rename'), self)
        self.action_rename.setShortcut('F2')
        self.action_rename.triggered.connect(self._on_rename_triggered)
        self.action_rename.setIcon(QtGui.QIcon.fromTheme('edit-rename'))
        self.addAction(self.action_rename)
        # Delete
        self.action_delete = QtWidgets.QAction(('&Delete'), self)
        self.action_delete.setShortcut(QtGui.QKeySequence.Delete)
        self.action_delete.setIcon(icon(
            'edit-delete', ':/pyqode-icons/rc/edit-delete.png', 'fa.remove'))
        self.action_delete.triggered.connect(self._on_delete_triggered)
        self.addAction(self.action_delete)
        self.addSeparator()

        text = 'Show in %s' % self.get_file_explorer_name()
        action = self.action_show_in_explorer = self.addAction(text)
        action.setIcon(QtGui.QIcon.fromTheme('system-file-manager'))
        action.triggered.connect(self._on_show_in_explorer_triggered)
        self._action_show_in_explorer = action

    def update_show_in_explorer_action(self):
        self.action_show_in_explorer.setText(
            ('Show in %s') % self.get_file_explorer_name())

    def _on_set_as_root_triggered(self):
        path = self.tree_view.helper.get_current_path()
        self.tree_view._on_set_as_root_triggered()

    def _on_delete_triggered(self):
        self.tree_view.helper.delete()

    def _on_copy_path_triggered(self):
        self.tree_view.helper.copy_path_to_clipboard()

    def _on_rename_triggered(self):
        self.tree_view.helper.rename()

    def _on_create_directory_triggered(self):
        self.tree_view.helper.create_directory()

    @classmethod
    def get_linux_file_explorer(cls):
        if cls._explorer is None:
            try:
                output = subprocess.check_output(
                    ['xdg-mime', 'query', 'default', 'inode/directory']).decode()
            except subprocess.CalledProcessError:
                output = ''
            if output:
                explorer = output.splitlines()[0].replace(
                    '.desktop', '').replace('-folder-handler', '').split(
                        '.')[-1].lower()
                FileSystemContextMenu._explorer = explorer
                return explorer
            return ''
        else:
            return cls._explorer

    @classmethod
    def get_file_explorer_name(cls):
        system = platform.system()
        if system == 'Darwin':
            pgm = 'finder'
        elif system == 'Windows':
            pgm = 'explorer'
        else:
            pgm = cls.get_file_explorer_command().split(' ')[0]
            if os.path.isabs(pgm):
                pgm = os.path.split(pgm)[1]
        return pgm.capitalize()

    def _on_show_in_explorer_triggered(self):
        path = self.tree_view.helper.get_current_path()
        self.show_in_explorer(path, self.tree_view)

    @classmethod
    def get_file_explorer_command(cls):
        if cls._command is None:
            system = platform.system()
            if system == 'Linux':
                explorer = cls.get_linux_file_explorer()
                if explorer in ['nautilus', 'dolphin']:
                    explorer_cmd = '%s --select %s' % (explorer, '%s')
                else:
                    explorer_cmd = '%s %s' % (explorer, '%s')
            elif system == 'Windows':
                explorer_cmd = 'explorer /select,%s'
            elif system == 'Darwin':
                explorer_cmd = 'open -R %s'
            cls._command = explorer_cmd
            return explorer_cmd
        else:
            return cls._command

    @classmethod
    def set_file_explorer_command(cls, command):
        pgm = command.split(' ')[0]
        if os.path.isabs(pgm):
            pgm = os.path.split(pgm)[1]
        cls._explorer = pgm
        cls._command = command

    @classmethod
    def show_in_explorer(cls, path, parent):
        try:
            cmd = cls.get_file_explorer_command() % os.path.normpath(path)
            args = cmd.split(' ')
            subprocess.Popen(args)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                parent, ('Open in explorer'),
                ('Failed to open file in explorer.\n\n%s') % str(e))

class FSContextMenu(FileSystemContextMenu):
    def __init__(self, app):
        super().__init__()
        self.app = app
