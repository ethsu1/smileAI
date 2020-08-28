var HtmlWebpackPlugin =  require('html-webpack-plugin');
var path = require('path');
const rules = [
            {test : /\.(js)$/, use:'babel-loader', exclude: /node_modules/ },
            {test : /\.css$/, use:['style-loader', 'css-loader']},
            {test: /\.(jpe?g|png|gif|woff|woff2|eot|ttf|svg)(\?[a-z0-9=.]+)?$/, loader: 'url-loader?limit=100000'}
        ]
module.exports = {
    entry: './src/index.js',
    module: {rules},
    output: {
        path: path.resolve(__dirname, 'dist'),
        //publicPath: __dirname + '/src/frontend/public',
        filename: 'index_bundle.js',
    },
    plugins: [
        new HtmlWebpackPlugin({
        template: './src/index.html'
        })
    ]
};