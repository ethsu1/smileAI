import React from 'react';
import {Helmet} from 'react-helmet';
import Camera from './camera.js'

function App() {
  return (
    <div>
      <Helmet>
        <style>{'body { background-color: #98FB98; }'}</style>
      </Helmet>
      <Camera/>
    </div>
  );
}

export default App;
